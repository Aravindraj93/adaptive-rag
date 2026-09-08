"""Segment-based sparse indexing for cheap append/delete and explicit compaction."""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from functools import wraps
from pathlib import Path
from typing import Iterable

from ..filtering import MetadataFilter
from ..models import Chunk, Query, SearchResult
from ..persistence import IndexFormatError, read_manifest, write_manifest
from ..tokenization import NormalizedTokenizer
from ..writer_lock import writer_lock
from ..reader_leases import reader_snapshot, lease_guard, pinned_segments
from .mmap_bm25 import MMapBM25Retriever


def serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with writer_lock(self.path):
            self._recover()
            return method(self, *args, **kwargs)
    return call


class SegmentedBM25Retriever:
    """Search immutable BM25 segments and fuse their local rankings with RRF."""

    MANIFEST = "segments.json"

    @reader_snapshot
    def __init__(self, path: str | Path, *, verify_checksums: bool = True) -> None:
        self.path = Path(path)
        payload = read_manifest(self.path / self.MANIFEST)
        if payload.get("engine") != "segmented-bm25" or payload.get("engine_version") != 1:
            raise IndexFormatError("unsupported segmented index engine")
        segments = payload.get("segments")
        deleted_ids = payload.get("deleted_ids", [])
        if not isinstance(segments, list) or not segments or not isinstance(deleted_ids, list):
            raise IndexFormatError("invalid segmented index manifest")
        self.deleted_ids = frozenset(str(value) for value in deleted_ids)
        self._segments = []
        for name in segments:
            if not isinstance(name, str) or not name.startswith('segment-') or Path(name).name != name or '/' in name or '\\' in name:
                raise IndexFormatError('invalid segment path')
            if (self.path / name).is_symlink() or (self.path / name).resolve().parent != self.path.resolve():
                raise IndexFormatError('segment path escapes index')
            self._segments.append(MMapBM25Retriever(self.path / name, verify_checksums=verify_checksums))
        self._closed = False

    def search(
        self,
        query: str | Query,
        *,
        top_k: int = 5,
        where: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        if self._closed:
            raise RuntimeError("segmented index is closed")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for segment in self._segments:
            live_rank = 0
            for result in segment.search(query, top_k=top_k + len(self.deleted_ids), where=where):
                if result.chunk.id in self.deleted_ids:
                    continue
                live_rank += 1
                if live_rank > top_k:
                    break
                chunk_id = result.chunk.id
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (60 + live_rank)
                chunks[chunk_id] = result.chunk
        ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
        return [
            SearchResult(chunks[chunk_id], scores[chunk_id], rank, "segmented-bm25")
            for rank, chunk_id in enumerate(ranked, start=1)
        ]

    def iter_chunks(self):
        if self._closed:
            raise RuntimeError("segmented index is closed")
        seen: set[str] = set()
        for segment in self._segments:
            for chunk in segment.iter_chunks():
                if chunk.id not in self.deleted_ids and chunk.id not in seen:
                    seen.add(chunk.id)
                    yield chunk

    def close(self) -> None:
        for segment in getattr(self, '_segments', []):
            segment.close()
        if getattr(self, '_lease', None) is not None:
            self._lease.__exit__(None, None, None)
            self._lease = None
        self._closed = True

    def __enter__(self) -> "SegmentedBM25Retriever":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class SegmentedBM25Index:
    """Manage immutable segments, tombstones, and full compaction."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def create(
        cls,
        path: str | Path,
        chunks: Iterable[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: NormalizedTokenizer | None = None,
    ) -> "SegmentedBM25Index":
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"segmented index already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        try:
            active_tokenizer = tokenizer or NormalizedTokenizer()
            MMapBM25Retriever.build(
                temporary / "segment-000000",
                chunks,
                k1=k1,
                b=b,
                tokenizer=active_tokenizer,
            ).close()
            write_manifest(
                temporary / SegmentedBM25Retriever.MANIFEST,
                {
                    "engine": "segmented-bm25",
                    "engine_version": 1,
                    "segments": ["segment-000000"],
                    "deleted_ids": [],
                    "configuration": {
                        "k1": k1,
                        "b": b,
                        "tokenizer": active_tokenizer.to_dict(),
                    },
                },
            )
            os.replace(temporary, destination)
            return cls(destination)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    @serialized
    def append(self, chunks: Iterable[Chunk]) -> int:
        additions = list(chunks)
        if not additions:
            return 0
        ids = [chunk.id for chunk in additions]
        if len(ids) != len(set(ids)):
            raise ValueError("appended chunk ids must be unique")
        payload = self._payload()
        with SegmentedBM25Retriever(self.path) as retriever:
            existing_ids = {chunk.id for chunk in retriever.iter_chunks()}
        unavailable_ids = existing_ids | set(payload["deleted_ids"])
        conflicts = unavailable_ids & set(ids)
        if conflicts:
            raise ValueError(f"chunk ids already exist: {', '.join(sorted(conflicts))}")
        configuration = payload["configuration"]
        segment_name = 'segment-' + uuid.uuid4().hex
        MMapBM25Retriever.build(
            self.path / segment_name,
            additions,
            k1=float(configuration["k1"]),
            b=float(configuration["b"]),
            tokenizer=NormalizedTokenizer.from_dict(configuration["tokenizer"]),
        ).close()
        payload["segments"].append(segment_name)
        self._publish(payload)
        return len(additions)

    @serialized
    def delete(self, chunk_ids: Iterable[str]) -> int:
        requested = set(chunk_ids)
        payload = self._payload()
        with SegmentedBM25Retriever(self.path) as retriever:
            live_ids = {chunk.id for chunk in retriever.iter_chunks()}
        deleted = requested & live_ids
        if live_ids <= deleted:
            raise ValueError("an index must retain at least one chunk")
        payload["deleted_ids"] = sorted(set(payload["deleted_ids"]) | deleted)
        self._publish(payload)
        return len(deleted)

    @serialized
    def compact(self) -> int:
        payload = self._payload()
        configuration = payload["configuration"]
        with SegmentedBM25Retriever(self.path) as retriever:
            live_chunks = list(retriever.iter_chunks())
        segment_name = 'segment-' + uuid.uuid4().hex
        MMapBM25Retriever.build(
            self.path / segment_name,
            live_chunks,
            k1=float(configuration["k1"]),
            b=float(configuration["b"]),
            tokenizer=NormalizedTokenizer.from_dict(configuration["tokenizer"]),
        ).close()
        payload['segments'] = [segment_name]
        payload['deleted_ids'] = []
        self._publish(payload)
        return len(live_chunks)

    def _publish(self, payload):
        journal = self.path / 'pending.json'
        write_manifest(journal, payload)
        write_manifest(self.path / SegmentedBM25Retriever.MANIFEST, payload)
        journal.unlink()

    @serialized
    def cleanup(self, *, dry_run=True):
        """Reclaim superseded segments unreferenced by manifests or live readers.

        All readers must be v0.8+ lease-aware. Dry-run is the default.
        """
        with lease_guard(self.path) as root:
            protected = set(self._payload()['segments']) | pinned_segments(root)
            candidates = []
            for segment in self.path.iterdir():
                if not segment.name.startswith('segment-') or segment.name in protected:
                    continue
                if not segment.is_dir() or segment.is_symlink() or segment.resolve().parent != self.path.resolve():
                    continue
                if not (segment / 'manifest.json').is_file():
                    continue
                payload = read_manifest(segment / 'manifest.json')
                if payload.get('engine') != 'mmap-bm25':
                    continue
                candidates.append(segment)
            if not dry_run:
                for segment in candidates:
                    shutil.rmtree(segment)
            return sorted(segment.name for segment in candidates)

    def _recover(self):
        journal = self.path / 'pending.json'
        if not journal.exists():
            return False
        payload = read_manifest(journal)
        if payload.get('engine') != 'segmented-bm25':
            raise IndexFormatError('invalid pending publication')
        for name in payload['segments']:
            if not isinstance(name, str) or Path(name).name != name or '/' in name or '\\' in name:
                raise IndexFormatError('invalid pending segment path')
            with MMapBM25Retriever(self.path / name):
                pass
        write_manifest(self.path / SegmentedBM25Retriever.MANIFEST, payload)
        journal.unlink()
        return True

    def recover(self):
        """Finish a journaled publication; also called before every mutation."""
        with writer_lock(self.path):
            return self._recover()

    def _payload(self) -> dict:
        payload = read_manifest(self.path / SegmentedBM25Retriever.MANIFEST)
        if payload.get("engine") != "segmented-bm25":
            raise IndexFormatError("not a segmented BM25 index")
        return payload
