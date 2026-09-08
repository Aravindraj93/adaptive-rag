"""Binary memory-mapped exact dense retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import shutil
import struct
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

from ..filtering import MetadataFilter
from ..models import Chunk, Query, RetrievalStats, SearchResult
from ..persistence import IndexFormatError, read_manifest, write_manifest, validate_ranges
from ..telemetry import TelemetryCollector
from .dense import Embedder, _unit


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MMapDenseRetriever:
    """Read-only exact cosine search over float32 memory-mapped vectors."""

    MANIFEST = "manifest.json"
    CHUNKS = "chunks.jsonl"
    VECTORS = "vectors.f32"

    def __init__(
        self,
        path: str | Path,
        *,
        embedder: Embedder,
        telemetry: TelemetryCollector | None = None,
        verify_checksums: bool = True,
    ) -> None:
        self.path = Path(path)
        payload = read_manifest(self.path / self.MANIFEST)
        if payload.get("engine") != "mmap-dense" or payload.get("engine_version") != 1:
            raise IndexFormatError("unsupported memory-mapped dense engine")
        self._records = payload.get("chunks")
        if not isinstance(self._records, list) or not self._records:
            raise IndexFormatError("invalid memory-mapped dense chunk records")
        self.dimensions = payload.get('dimensions')
        if type(self.dimensions) is not int or self.dimensions < 1:
            raise IndexFormatError('dense dimensions must be a positive integer')
        self.min_score = payload.get("min_score", 0.0)
        self.embedder = embedder
        self.telemetry = telemetry or TelemetryCollector()
        self.last_stats: RetrievalStats | None = None
        self._closed = True
        chunks_path = self.path / self.CHUNKS
        vectors_path = self.path / self.VECTORS
        if verify_checksums:
            expected = payload.get("file_checksums")
            if not isinstance(expected, dict):
                raise IndexFormatError("invalid dense file checksums")
            if expected.get(self.CHUNKS) != _checksum(chunks_path):
                raise IndexFormatError("dense chunk checksum mismatch")
            if expected.get(self.VECTORS) != _checksum(vectors_path):
                raise IndexFormatError("dense vector checksum mismatch")
        try:
            self._chunks_file = chunks_path.open("rb")
            self._vectors_file = vectors_path.open("rb")
            self._chunks_map = mmap.mmap(self._chunks_file.fileno(), 0, access=mmap.ACCESS_READ)
            self._vectors_map = mmap.mmap(self._vectors_file.fileno(), 0, access=mmap.ACCESS_READ)
            validate_ranges(self._records, len(self._chunks_map))
            if len(self._vectors_map) != len(self._records) * self.dimensions * 4:
                raise IndexFormatError("dense vector file size mismatch")
            self._closed = False
        except Exception:
            self.close()
            raise

    def __len__(self) -> int:
        return len(self._records)

    @property
    def closed(self) -> bool:
        return self._closed

    @classmethod
    def build(
        cls,
        path: str | Path,
        chunks: Iterable[Chunk],
        *,
        embedder: Embedder,
        batch_size: int = 64,
        min_score: float | None = 0.0,
    ) -> "MMapDenseRetriever":
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"index already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        try:
            records: list[dict[str, int]] = []
            seen_ids: set[str] = set()
            dimensions: int | None = None
            with (temporary / cls.CHUNKS).open("wb") as chunk_stream, (
                temporary / cls.VECTORS
            ).open("wb") as vector_stream:
                batch: list[Chunk] = []

                def write_batch() -> None:
                    nonlocal dimensions
                    if not batch:
                        return
                    vectors = list(embedder([chunk.text for chunk in batch]))
                    if len(vectors) != len(batch):
                        raise ValueError("embedder returned a different number of vectors than texts")
                    for chunk, raw_vector in zip(batch, vectors):
                        vector = _unit(raw_vector)
                        if dimensions is None:
                            dimensions = len(vector)
                        elif len(vector) != dimensions:
                            raise ValueError("all embeddings must have the same dimensions")
                        encoded = json.dumps(
                            {
                                "id": chunk.id,
                                "document_id": chunk.document_id,
                                "text": chunk.text,
                                "metadata": dict(chunk.metadata),
                                "position": chunk.position,
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        offset = chunk_stream.tell()
                        chunk_stream.write(encoded + b"\n")
                        records.append({"offset": offset, "size": len(encoded)})
                        vector_stream.write(struct.pack(f"<{dimensions}f", *vector))

                for chunk in chunks:
                    if chunk.id in seen_ids:
                        raise ValueError(f"duplicate chunk id: {chunk.id}")
                    seen_ids.add(chunk.id)
                    batch.append(chunk)
                    if len(batch) == batch_size:
                        write_batch()
                        batch.clear()
                write_batch()
            if not records or dimensions is None:
                raise ValueError("at least one chunk is required")
            write_manifest(
                temporary / cls.MANIFEST,
                {
                    "engine": "mmap-dense",
                    "engine_version": 1,
                    "dimensions": dimensions,
                    "min_score": min_score,
                    "chunks": records,
                    "file_checksums": {
                        cls.CHUNKS: _checksum(temporary / cls.CHUNKS),
                        cls.VECTORS: _checksum(temporary / cls.VECTORS),
                    },
                },
            )
            os.replace(temporary, destination)
            return cls(destination, embedder=embedder)
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
            raise RuntimeError("memory-mapped dense index is closed")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        started = perf_counter()
        text = query.text if isinstance(query, Query) else query
        vectors = list(self.embedder([text]))
        if len(vectors) != 1:
            raise ValueError("embedder must return exactly one query vector")
        query_vector = _unit(vectors[0])
        if len(query_vector) != self.dimensions:
            raise ValueError("query embedding dimensions do not match the index")
        assert self._vectors_map is not None
        scored: list[tuple[float, int, Chunk]] = []
        vector_format = f"<{self.dimensions}f"
        vector_size = self.dimensions * 4
        for ordinal in range(len(self)):
            chunk = self._read_chunk(ordinal)
            if where is not None and not where.matches(chunk.metadata):
                continue
            vector = struct.unpack_from(vector_format, self._vectors_map, ordinal * vector_size)
            score = sum(left * right for left, right in zip(query_vector, vector))
            if self.min_score is None or score > self.min_score:
                scored.append((score, ordinal, chunk))
        scored.sort(key=lambda item: (-item[0], item[2].id))
        results = [
            SearchResult(chunk, score, rank, "mmap-dense")
            for rank, (score, _, chunk) in enumerate(scored[:top_k], start=1)
        ]
        latency_ms = (perf_counter() - started) * 1000
        self.last_stats = RetrievalStats(1, len(scored), len(results), latency_ms, "mmap-dense")
        self.telemetry.record(
            "mmap-dense.search", latency_ms, candidates=len(scored), returned=len(results)
        )
        return results

    def _read_chunk(self, ordinal: int) -> Chunk:
        record = self._records[ordinal]
        assert self._chunks_map is not None
        value = json.loads(self._chunks_map[record["offset"] : record["offset"] + record["size"]])
        return Chunk(
            value["id"], value["document_id"], value["text"], value.get("metadata", {}),
            int(value.get("position", 0)),
        )

    def close(self) -> None:
        for attribute in ("_chunks_map", "_vectors_map", "_chunks_file", "_vectors_file"):
            resource = getattr(self, attribute, None)
            if resource is not None:
                resource.close()
                setattr(self, attribute, None)
        self._closed = True

    def __enter__(self) -> "MMapDenseRetriever":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
