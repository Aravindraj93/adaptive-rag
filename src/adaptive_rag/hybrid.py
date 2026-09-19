"""Unified high-level hybrid retrieval facade combining lexical, dense, and rank-fused search."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

from .cache import CachedRetriever
from .composition import ReciprocalRankFusionRetriever
from .filtering import MetadataFilter
from .models import Chunk, Query, SearchResult
from .retrievers.base import Retriever
from .retrievers.bm25 import BM25Retriever
from .retrievers.dense import DenseRetriever, Embedder
from .tokenization import NormalizedTokenizer


class HybridRetriever:
    """Convenience facade combining BM25, Dense, and rank-fused hybrid retrieval."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        split_identifiers: bool = True,
        anchor_boost: float = 1.0,
        anchor_gap_threshold: float = 0.25,
        weights: Sequence[float] = (1.0, 1.0),
        rank_constant: int = 60,
        candidate_multiplier: int = 3,
        revision: Callable[[], str] | None = None,
        scope: str = "default",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.tokenizer = NormalizedTokenizer(split_identifiers=split_identifiers)
        self.sparse = BM25Retriever(k1=k1, b=b, tokenizer=self.tokenizer)
        self.dense = DenseRetriever(embedder)
        self.fusion = ReciprocalRankFusionRetriever(
            [self.sparse, self.dense],
            weights=weights,
            rank_constant=rank_constant,
            candidate_multiplier=candidate_multiplier,
            anchor_boost=anchor_boost,
            anchor_gap_threshold=anchor_gap_threshold,
        )
        if revision is not None:
            self._backend: Retriever = CachedRetriever(
                self.fusion, revision=revision, scope=scope
            )
        else:
            self._backend = self.fusion

    def __len__(self) -> int:
        return len(self.sparse)

    def add(self, chunks: Iterable[Chunk]) -> None:
        """Add chunks to both sparse and dense indexes."""
        chunk_list = list(chunks)
        self.sparse.add(chunk_list)
        self.dense.add(chunk_list)

    def search(
        self,
        query: str | Query,
        *,
        top_k: int = 5,
        where: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """Search across both lexical and dense indexes with rank fusion."""
        return self._backend.search(query, top_k=top_k)
