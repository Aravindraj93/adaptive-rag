"""Serialized background coordination for segmented index mutations."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from .models import Chunk
from .retrievers.segmented_bm25 import SegmentedBM25Index


class AsyncSegmentCoordinator:
    """Run append, delete, and compaction jobs in one ordered worker thread."""

    def __init__(self, path: str | Path) -> None:
        self.manager = SegmentedBM25Index(path)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="adaptive-rag-index")
        self._closed = False

    def append(self, chunks: Iterable[Chunk]) -> Future[int]:
        additions = tuple(chunks)
        return self._submit(self.manager.append, additions)

    def delete(self, chunk_ids: Iterable[str]) -> Future[int]:
        ids = tuple(chunk_ids)
        return self._submit(self.manager.delete, ids)

    def compact(self) -> Future[int]:
        return self._submit(self.manager.compact)

    def _submit(self, function, *args) -> Future[int]:
        if self._closed:
            raise RuntimeError("background index coordinator is closed")
        return self._executor.submit(function, *args)

    def close(self, *, wait: bool = True) -> None:
        if not self._closed:
            self._executor.shutdown(wait=wait, cancel_futures=not wait)
            self._closed = True

    def __enter__(self) -> "AsyncSegmentCoordinator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

