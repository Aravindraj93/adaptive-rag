"""Generic fusion and selective reranking wrappers."""

from __future__ import annotations

from typing import Callable, Sequence

from .models import Chunk, Query, SearchResult
from .retrievers.base import Retriever

Reranker = Callable[[str, Sequence[Chunk]], Sequence[float]]


class ReciprocalRankFusionRetriever:
    """Fuse ranked lists without requiring comparable source scores."""

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        rank_constant: int = 60,
        candidate_multiplier: int = 3,
        weights: Sequence[float] | None = None,
    ) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        if rank_constant < 1 or candidate_multiplier < 1:
            raise ValueError("rank_constant and candidate_multiplier must be positive")
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError("weights must match the number of retrievers")
        self.retrievers = tuple(retrievers)
        self.rank_constant = rank_constant
        self.candidate_multiplier = candidate_multiplier
        self.weights = tuple(weights or [1.0] * len(retrievers))

    def search(self, query: str | Query, *, top_k: int = 5) -> list[SearchResult]:
        return self._search_with_prefetched(query, top_k=top_k)

    def _search_with_prefetched(
        self, query: str | Query, *, top_k: int,
        primary_results: Sequence[SearchResult] | None = None,
    ) -> list[SearchResult]:
        # Internal contract: results belong to retrievers[0], this exact query
        # and snapshot, at depth top_k * candidate_multiplier. Never cached.
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        sources: dict[str, set[str]] = {}
        candidate_k = top_k * self.candidate_multiplier
        for index, (retriever, weight) in enumerate(zip(self.retrievers, self.weights)):
            results = (primary_results if index == 0 and primary_results is not None
                       else retriever.search(query, top_k=candidate_k))
            for result in results:
                chunk_id = result.chunk.id
                scores[chunk_id] = scores.get(chunk_id, 0.0) + (
                    weight / (self.rank_constant + result.rank)
                )
                chunks.setdefault(chunk_id, result.chunk)
                sources.setdefault(chunk_id, set()).add(result.source)
        ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
        return [
            SearchResult(
                chunks[chunk_id],
                scores[chunk_id],
                rank,
                "rrf:" + "+".join(sorted(sources[chunk_id])),
            )
            for rank, chunk_id in enumerate(ranked, start=1)
        ]


class SelectiveRerankingRetriever:
    """Invoke an application-supplied reranker only for ambiguous top results."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        *,
        ambiguity_threshold: float = 0.15,
        candidate_multiplier: int = 3,
    ) -> None:
        if not 0 <= ambiguity_threshold <= 1 or candidate_multiplier < 1:
            raise ValueError("invalid reranking settings")
        self.retriever = retriever
        self.reranker = reranker
        self.ambiguity_threshold = ambiguity_threshold
        self.candidate_multiplier = candidate_multiplier
        self.last_reranked = False

    def search(self, query: str | Query, *, top_k: int = 5) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        candidates = list(
            self.retriever.search(query, top_k=top_k * self.candidate_multiplier)
        )
        self.last_reranked = self._ambiguous(candidates)
        if not self.last_reranked:
            return [
                SearchResult(result.chunk, result.score, rank, result.source)
                for rank, result in enumerate(candidates[:top_k], start=1)
            ]
        text = query.text if isinstance(query, Query) else query
        scores = list(self.reranker(text, [result.chunk for result in candidates]))
        if len(scores) != len(candidates):
            raise ValueError("reranker returned a different number of scores than chunks")
        rescored = sorted(
            zip((float(score) for score in scores), candidates),
            key=lambda item: (-item[0], item[1].chunk.id),
        )[:top_k]
        return [
            SearchResult(result.chunk, score, rank, f"reranked:{result.source}")
            for rank, (score, result) in enumerate(rescored, start=1)
        ]

    def _ambiguous(self, candidates: list[SearchResult]) -> bool:
        if len(candidates) < 2 or candidates[0].score <= 0:
            return False
        relative_gap = (candidates[0].score - candidates[1].score) / candidates[0].score
        return relative_gap <= self.ambiguity_threshold
