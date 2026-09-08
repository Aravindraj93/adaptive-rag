"""Immutable memory-mapped BM25 index for larger CPU-first corpora."""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import shutil
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Iterable

from ..filtering import MetadataFilter
from ..models import Chunk, Query, RetrievalStats, SearchResult
from ..persistence import IndexFormatError, read_manifest, write_manifest, validate_ranges
from ..telemetry import TelemetryCollector
from ..tokenization import NormalizedTokenizer


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MMapBM25Retriever:
    """Read-only BM25 index whose chunks and postings stay memory-mapped."""

    MANIFEST = "manifest.json"
    CHUNKS = "chunks.jsonl"
    POSTINGS = "postings.jsonl"
    FACETS = "facets.sqlite"

    def __init__(
        self,
        path: str | Path,
        *,
        telemetry: TelemetryCollector | None = None,
        verify_checksums: bool = True,
    ) -> None:
        self.path = Path(path)
        payload = read_manifest(self.path / self.MANIFEST)
        if payload.get("engine") != "mmap-bm25" or payload.get("engine_version") != 1:
            raise IndexFormatError("unsupported memory-mapped index engine")
        configuration = payload.get("configuration")
        if not isinstance(configuration, dict):
            raise IndexFormatError("invalid memory-mapped index configuration")
        try:
            self.k1 = float(configuration['k1'])
            self.b = float(configuration['b'])
        except (KeyError, TypeError, ValueError) as error:
            raise IndexFormatError('invalid BM25 parameters') from error
        if not math.isfinite(self.k1) or self.k1 <= 0 or not math.isfinite(self.b) or not 0 <= self.b <= 1:
            raise IndexFormatError('invalid BM25 parameter ranges')
        tokenizer_config = configuration.get("tokenizer")
        if not isinstance(tokenizer_config, dict):
            raise IndexFormatError("invalid tokenizer configuration")
        self.tokenizer = NormalizedTokenizer.from_dict(tokenizer_config)
        self._chunk_records = payload.get("chunks")
        self._term_records = payload.get("terms")
        if not isinstance(self._chunk_records, list) or not isinstance(self._term_records, dict):
            raise IndexFormatError("invalid memory-mapped index records")
        self._total_terms = int(payload.get("total_terms", 0))
        self.telemetry = telemetry or TelemetryCollector()
        self.last_stats: RetrievalStats | None = None
        self._closed = True
        self._chunks_file = None
        self._postings_file = None
        self._chunks_map = None
        self._postings_map = None
        self._facets = None
        self._facet_lock = Lock()

        chunks_path = self.path / self.CHUNKS
        postings_path = self.path / self.POSTINGS
        facets_path = self.path / self.FACETS
        if not isinstance(payload.get('file_checksums', {}), dict):
            raise IndexFormatError('invalid checksum metadata')
        self._has_facets = self.FACETS in payload.get('file_checksums', {})
        if verify_checksums:
            expected = payload.get("file_checksums", {})
            if not isinstance(expected, dict):
                raise IndexFormatError("invalid file checksum records")
            if expected.get(self.CHUNKS) != _file_checksum(chunks_path):
                raise IndexFormatError("chunk data checksum mismatch")
            if expected.get(self.POSTINGS) != _file_checksum(postings_path):
                raise IndexFormatError("posting data checksum mismatch")
            if self._has_facets and expected.get(self.FACETS) != _file_checksum(facets_path):
                raise IndexFormatError("facet data checksum mismatch")
        try:
            self._chunks_file = chunks_path.open("rb")
            self._postings_file = postings_path.open("rb")
            self._chunks_map = mmap.mmap(self._chunks_file.fileno(), 0, access=mmap.ACCESS_READ)
            self._postings_map = mmap.mmap(self._postings_file.fileno(), 0, access=mmap.ACCESS_READ)
            validate_ranges(self._chunk_records, len(self._chunks_map))
            validate_ranges(self._term_records.values(), len(self._postings_map))
            facet_uri = facets_path.resolve().as_uri() + "?mode=ro"
            if self._has_facets:
                self._facets = sqlite3.connect(facet_uri, uri=True, check_same_thread=False)
            self._closed = False
        except Exception:
            self.close()
            raise

    def __len__(self) -> int:
        return len(self._chunk_records)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def average_document_length(self) -> float:
        return self._total_terms / len(self) if len(self) else 0.0

    @classmethod
    def build(
        cls,
        path: str | Path,
        chunks: Iterable[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: NormalizedTokenizer | None = None,
        overwrite: bool = False,
    ) -> "MMapBM25Retriever":
        """Build an immutable index in a new directory, then open it."""

        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be positive and b must be between 0 and 1")
        destination = Path(path)
        if destination.exists():
            if not overwrite:
                raise FileExistsError(f"index already exists: {destination}")
            if not destination.is_dir() or not (destination / cls.MANIFEST).is_file():
                raise ValueError("overwrite target is not an adaptive-rag mmap index")
        active_tokenizer = tokenizer or NormalizedTokenizer()
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=parent))
        try:
            chunk_records: list[dict[str, object]] = []
            total_terms = 0
            seen_ids: set[str] = set()
            staging_path = temporary / "postings.sqlite"
            staging = sqlite3.connect(staging_path)
            try:
                staging.execute("PRAGMA journal_mode=OFF")
                staging.execute("PRAGMA synchronous=OFF")
                staging.execute("PRAGMA temp_store=FILE")
                staging.execute("PRAGMA cache_size=-2048")
                staging.execute(
                    "CREATE TABLE postings (term TEXT NOT NULL, ordinal INTEGER NOT NULL, frequency INTEGER NOT NULL)"
                )
                staging.execute(
                    "CREATE TABLE facets (key TEXT NOT NULL, value TEXT NOT NULL, ordinal INTEGER NOT NULL)"
                )
                with (temporary / cls.CHUNKS).open("wb") as stream:
                    for ordinal, chunk in enumerate(chunks):
                        if chunk.id in seen_ids:
                            raise ValueError(f"duplicate chunk id: {chunk.id}")
                        seen_ids.add(chunk.id)
                        encoded = _json_bytes(
                            {
                                "id": chunk.id,
                                "document_id": chunk.document_id,
                                "text": chunk.text,
                                "metadata": dict(chunk.metadata),
                                "position": chunk.position,
                            }
                        )
                        offset = stream.tell()
                        stream.write(encoded + b"\n")
                        frequencies = Counter(active_tokenizer(chunk.text))
                        length = sum(frequencies.values())
                        total_terms += length
                        chunk_records.append(
                            {"offset": offset, "size": len(encoded), "term_count": length, "id": chunk.id}
                        )
                        staging.executemany(
                            "INSERT INTO postings VALUES (?, ?, ?)",
                            ((term, ordinal, frequency) for term, frequency in frequencies.items()),
                        )
                        staging.executemany(
                            "INSERT INTO facets VALUES (?, ?, ?)",
                            (
                                (
                                    str(key),
                                    json.dumps(
                                        value,
                                        ensure_ascii=False,
                                        sort_keys=True,
                                        separators=(",", ":"),
                                        allow_nan=False,
                                    ),
                                    ordinal,
                                )
                                for key, value in chunk.metadata.items()
                            ),
                        )
                if not chunk_records:
                    raise ValueError("at least one chunk is required")
                staging.commit()

                term_records: dict[str, dict[str, int]] = {}
                with (temporary / cls.POSTINGS).open("wb") as stream:
                    current_term: str | None = None
                    current_postings: list[list[int]] = []

                    def flush_postings() -> None:
                        if current_term is None:
                            return
                        encoded = _json_bytes(current_postings)
                        offset = stream.tell()
                        stream.write(encoded + b"\n")
                        term_records[current_term] = {
                            "offset": offset,
                            "size": len(encoded),
                            "document_frequency": len(current_postings),
                        }

                    cursor = staging.execute(
                        "SELECT term, ordinal, frequency FROM postings ORDER BY term, ordinal"
                    )
                    for term, ordinal, frequency in cursor:
                        if current_term is not None and term != current_term:
                            flush_postings()
                            current_postings = []
                        current_term = term
                        current_postings.append([ordinal, frequency])
                    flush_postings()
                staging.execute("DROP TABLE postings")
                staging.execute("CREATE INDEX facets_key_value ON facets (key, value)")
                staging.execute("CREATE INDEX facets_key ON facets (key)")
                staging.commit()
                staging.execute("VACUUM")
            finally:
                staging.close()
            os.replace(staging_path, temporary / cls.FACETS)
            write_manifest(
                temporary / cls.MANIFEST,
                {
                    "engine": "mmap-bm25",
                    "engine_version": 1,
                    "configuration": {
                        "k1": k1,
                        "b": b,
                        "tokenizer": active_tokenizer.to_dict(),
                    },
                    "chunks": chunk_records,
                    "terms": term_records,
                    "total_terms": total_terms,
                    "file_checksums": {
                        cls.CHUNKS: _file_checksum(temporary / cls.CHUNKS),
                        cls.POSTINGS: _file_checksum(temporary / cls.POSTINGS),
                        cls.FACETS: _file_checksum(temporary / cls.FACETS),
                    },
                },
            )
            if destination.exists():
                backup = destination.with_name(destination.name + ".previous")
                if backup.exists():
                    raise FileExistsError(f"index backup already exists: {backup}")
                os.replace(destination, backup)
                try:
                    os.replace(temporary, destination)
                except Exception:
                    os.replace(backup, destination)
                    raise
                shutil.rmtree(backup)
            else:
                os.replace(temporary, destination)
            return cls(destination)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    def search(
        self,
        query: str | Query,
        *,
        top_k: int = 5,
        where: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        if self.closed:
            raise RuntimeError("memory-mapped index is closed")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        started = perf_counter()
        query_text = query.text if isinstance(query, Query) else query
        terms = self.tokenizer(query_text)
        scores: dict[int, float] = {}
        allowed_ordinals = self._filter_ordinals(where)
        average_length = self.average_document_length or 1.0
        assert self._postings_map is not None
        for term in set(terms):
            record = self._term_records.get(term)
            if not isinstance(record, dict):
                continue
            postings = json.loads(
                self._postings_map[record["offset"] : record["offset"] + record["size"]]
            )
            document_frequency = record["document_frequency"]
            inverse_document_frequency = math.log(
                1 + (len(self) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for ordinal, frequency in postings:
                if allowed_ordinals is not None and ordinal not in allowed_ordinals:
                    continue
                document_length = self._chunk_records[ordinal]["term_count"]
                normalization = frequency + self.k1 * (
                    1 - self.b + self.b * document_length / average_length
                )
                scores[ordinal] = scores.get(ordinal, 0.0) + (
                    inverse_document_frequency * frequency * (self.k1 + 1) / normalization
                )
        ranked = sorted(scores.items(), key=lambda item: (
            -item[1], self._chunk_records[item[0]].get('id') or self._read_chunk(item[0]).id
        ))[:top_k]

        results = [
            SearchResult(self._read_chunk(ordinal), score, rank, "mmap-bm25")
            for rank, (ordinal, score) in enumerate(ranked, start=1)
        ]
        latency_ms = (perf_counter() - started) * 1000
        self.last_stats = RetrievalStats(
            len(terms), len(scores), len(results), latency_ms, "mmap-bm25"
        )
        self.telemetry.record(
            "mmap-bm25.search",
            latency_ms,
            query_terms=len(terms),
            candidates=len(scores),
            returned=len(results),
        )
        return results

    def _read_chunk(self, ordinal: int) -> Chunk:
        record = self._chunk_records[ordinal]
        assert self._chunks_map is not None
        value = json.loads(self._chunks_map[record["offset"] : record["offset"] + record["size"]])
        return Chunk(
            value["id"],
            value["document_id"],
            value["text"],
            value.get("metadata", {}),
            int(value.get("position", 0)),
        )

    def _filter_ordinals(self, where: MetadataFilter | None) -> set[int] | None:
        if where is None or not (where.equals or where.any_of or where.exists):
            return None
        if self._facets is None:
            return {ordinal for ordinal in range(len(self)) if where.matches(self._read_chunk(ordinal).metadata)}
        selected: set[int] | None = None

        def intersect(values: set[int]) -> None:
            nonlocal selected
            selected = values if selected is None else selected & values

        with self._facet_lock:
            for key, value in where.equals.items():
                encoded = json.dumps(
                    value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                rows = self._facets.execute(
                    "SELECT ordinal FROM facets WHERE key = ? AND value = ?",
                    (key, encoded),
                )
                intersect({row[0] for row in rows})
            for key, choices in where.any_of.items():
                values: set[int] = set()
                for choice in choices:
                    encoded = json.dumps(
                        choice,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    rows = self._facets.execute(
                        "SELECT ordinal FROM facets WHERE key = ? AND value = ?",
                        (key, encoded),
                    )
                    values.update(row[0] for row in rows)
                intersect(values)
            for key in where.exists:
                rows = self._facets.execute(
                    "SELECT ordinal FROM facets WHERE key = ?", (key,)
                )
                intersect({row[0] for row in rows})
        return selected or set()

    def iter_chunks(self):
        """Yield chunks in stable index order without materializing the corpus."""

        if self.closed:
            raise RuntimeError("memory-mapped index is closed")
        for ordinal in range(len(self)):
            yield self._read_chunk(ordinal)

    def close(self) -> None:
        for attribute in (
            "_chunks_map",
            "_postings_map",
            "_chunks_file",
            "_postings_file",
            "_facets",
        ):
            resource = getattr(self, attribute, None)
            if resource is not None:
                resource.close()
                setattr(self, attribute, None)
        self._closed = True

    def __enter__(self) -> "MMapBM25Retriever":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
