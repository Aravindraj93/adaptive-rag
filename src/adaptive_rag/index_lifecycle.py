"""Safe lifecycle operations for immutable memory-mapped indexes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import Chunk
from .retrievers.mmap_bm25 import MMapBM25Retriever


@dataclass(frozen=True, slots=True)
class IndexUpdateReport:
    before: int
    added: int
    replaced: int
    deleted: int
    after: int


class MMapIndexManager:
    """Rebuild immutable indexes to apply upserts, deletions, or compaction."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def update(
        self,
        *,
        upserts: Iterable[Chunk] = (),
        delete_ids: Iterable[str] = (),
    ) -> IndexUpdateReport:
        with MMapBM25Retriever(self.path) as current:
            existing = {chunk.id: chunk for chunk in current.iter_chunks()}
            k1, b, tokenizer = current.k1, current.b, current.tokenizer
        before = len(existing)
        deletions = set(delete_ids)
        deleted = sum(chunk_id in existing for chunk_id in deletions)
        for chunk_id in deletions:
            existing.pop(chunk_id, None)
        additions = list(upserts)
        incoming_ids = [chunk.id for chunk in additions]
        if len(incoming_ids) != len(set(incoming_ids)):
            raise ValueError("upsert chunk ids must be unique")
        replaced = sum(chunk.id in existing for chunk in additions)
        for chunk in additions:
            existing[chunk.id] = chunk
        if not existing:
            raise ValueError("an index must retain at least one chunk")
        MMapBM25Retriever.build(
            self.path,
            existing.values(),
            k1=k1,
            b=b,
            tokenizer=tokenizer,
            overwrite=True,
        ).close()
        return IndexUpdateReport(before, len(additions) - replaced, replaced, deleted, len(existing))

    def append(self, chunks: Iterable[Chunk]) -> IndexUpdateReport:
        additions = list(chunks)
        with MMapBM25Retriever(self.path) as current:
            existing_ids = {chunk.id for chunk in current.iter_chunks()}
        conflicts = existing_ids & {chunk.id for chunk in additions}
        if conflicts:
            raise ValueError(f"chunk ids already exist: {', '.join(sorted(conflicts))}")
        return self.update(upserts=additions)

    def delete(self, chunk_ids: Iterable[str]) -> IndexUpdateReport:
        return self.update(delete_ids=chunk_ids)

    def compact(self) -> IndexUpdateReport:
        return self.update()

