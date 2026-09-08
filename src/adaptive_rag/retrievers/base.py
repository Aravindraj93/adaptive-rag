"""Protocol implemented by retrieval strategies."""

from __future__ import annotations

from typing import Iterable, Protocol, Sequence

from ..models import Chunk, Query, SearchResult


class Retriever(Protocol):
    def add(self, chunks: Iterable[Chunk]) -> None: ...

    def search(self, query: str | Query, *, top_k: int = 5) -> Sequence[SearchResult]: ...

