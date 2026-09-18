"""Generic fusion and selective reranking wrappers."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

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
        anchor_boost: float = 0.0,
        anchor_gap_threshold: float = 0.25,
    ) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        if rank_constant < 1 or candidate_multiplier < 1:
            raise ValueError("rank_constant and candidate_multiplier must be positive")
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError("weights must match the number of retrievers")
        if anchor_boost < 0:
            raise ValueError("anchor_boost must be non-negative")
        self.retrievers = tuple(retrievers)
        self.rank_constant = rank_constant
        self.candidate_multiplier = candidate_multiplier
        self.weights = tuple(weights or [1.0] * len(retrievers))
        self.anchor_boost = anchor_boost
        self.anchor_gap_threshold = anchor_gap_threshold

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
        primary_hits: Sequence[SearchResult] | None = None

        for index, (retriever, weight) in enumerate(zip(self.retrievers, self.weights)):
            results = (primary_results if index == 0 and primary_results is not None
                       else retriever.search(query, top_k=candidate_k))
            if index == 0:
                primary_hits = results
            for result in results:
                chunk_id = result.chunk.id
                scores[chunk_id] = scores.get(chunk_id, 0.0) + (
                    weight / (self.rank_constant + result.rank)
                )
                chunks.setdefault(chunk_id, result.chunk)
                sources.setdefault(chunk_id, set()).add(result.source)

        if self.anchor_boost > 0 and primary_hits:
            top_hit = primary_hits[0]
            if len(primary_hits) == 1:
                scores[top_hit.chunk.id] = scores.get(top_hit.chunk.id, 0.0) + self.anchor_boost
            elif len(primary_hits) > 1 and top_hit.score > 0:
                gap = (top_hit.score - primary_hits[1].score) / top_hit.score
                if gap >= self.anchor_gap_threshold:
                    scores[top_hit.chunk.id] = scores.get(top_hit.chunk.id, 0.0) + self.anchor_boost

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


class ScoreWeightedFusionRetriever:
    """Fuse multiple retrievers by normalizing and weighting their raw similarity scores."""

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        weights: Sequence[float] | None = None,
        candidate_multiplier: int = 3,
    ) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be positive")
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError("weights must match the number of retrievers")
        self.retrievers = tuple(retrievers)
        self.candidate_multiplier = candidate_multiplier
        self.weights = tuple(weights or [1.0] * len(retrievers))

    def search(self, query: str | Query, *, top_k: int = 5) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        candidate_k = top_k * self.candidate_multiplier
        combined_scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        sources: dict[str, set[str]] = {}

        for retriever, weight in zip(self.retrievers, self.weights):
            results = list(retriever.search(query, top_k=candidate_k))
            if not results:
                continue
            scores = [r.score for r in results]
            max_s = max(scores)
            min_s = min(scores)
            span = max_s - min_s if max_s > min_s else 1.0

            for result in results:
                cid = result.chunk.id
                norm = (result.score - min_s) / span if max_s > min_s else 1.0
                combined_scores[cid] = combined_scores.get(cid, 0.0) + (weight * norm)
                chunks.setdefault(cid, result.chunk)
                sources.setdefault(cid, set()).add(result.source)

        ranked = sorted(combined_scores, key=lambda cid: (-combined_scores[cid], cid))[:top_k]
        return [
            SearchResult(
                chunks[cid],
                combined_scores[cid],
                rank,
                "score_fusion:" + "+".join(sorted(sources[cid])),
            )
            for rank, cid in enumerate(ranked, start=1)
        ]


class RelationalExpansionRetriever:
    """Expand retrieved search results with related chunks joined by metadata references."""

    def __init__(
        self,
        retriever: Retriever,
        chunk_store: Mapping[str, Chunk],
        *,
        relation_keys: Sequence[str] = ("parent_id", "ref_ids", "trigger_ids", "related_ids"),
        include_reverse_references: bool = True,
        max_expansions_per_result: int = 3,
        discount_factor: float = 0.9,
    ) -> None:
        if max_expansions_per_result < 1:
            raise ValueError("max_expansions_per_result must be at least 1")
        if not 0 < discount_factor <= 1.0:
            raise ValueError("discount_factor must be in (0, 1]")
        self.retriever = retriever
        self.chunk_store = chunk_store
        self.relation_keys = tuple(relation_keys)
        self.include_reverse_references = include_reverse_references
        self.max_expansions_per_result = max_expansions_per_result
        self.discount_factor = discount_factor

        self._reverse_refs: dict[str, list[str]] = {}
        if self.include_reverse_references:
            for cid, chunk in self.chunk_store.items():
                for key in self.relation_keys:
                    val = chunk.metadata.get(key)
                    if isinstance(val, str) and val in self.chunk_store:
                        self._reverse_refs.setdefault(val, []).append(cid)
                    elif isinstance(val, (list, tuple, set)):
                        for item in val:
                            if isinstance(item, str) and item in self.chunk_store:
                                self._reverse_refs.setdefault(item, []).append(cid)

    def search(self, query: str | Query, *, top_k: int = 5) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        primary_results = list(self.retriever.search(query, top_k=top_k))
        expanded_results: list[SearchResult] = []
        seen_chunk_ids: set[str] = set()

        for result in primary_results:
            if result.chunk.id not in seen_chunk_ids:
                seen_chunk_ids.add(result.chunk.id)
                expanded_results.append(result)

            added_for_this = 0
            for key in self.relation_keys:
                if added_for_this >= self.max_expansions_per_result:
                    break
                ref = result.chunk.metadata.get(key)
                target_ids = (
                    [ref]
                    if isinstance(ref, str)
                    else (list(ref) if isinstance(ref, (list, tuple, set)) else [])
                )
                for target_id in target_ids:
                    if added_for_this >= self.max_expansions_per_result:
                        break
                    if target_id in self.chunk_store and target_id not in seen_chunk_ids:
                        seen_chunk_ids.add(target_id)
                        rel_chunk = self.chunk_store[target_id]
                        expanded_results.append(
                            SearchResult(
                                rel_chunk,
                                result.score * self.discount_factor,
                                len(expanded_results) + 1,
                                f"joined:{key}:{result.source}",
                            )
                        )
                        added_for_this += 1

            if self.include_reverse_references and added_for_this < self.max_expansions_per_result:
                for child_id in self._reverse_refs.get(result.chunk.id, ()):
                    if added_for_this >= self.max_expansions_per_result:
                        break
                    if child_id not in seen_chunk_ids and child_id in self.chunk_store:
                        seen_chunk_ids.add(child_id)
                        rel_chunk = self.chunk_store[child_id]
                        expanded_results.append(
                            SearchResult(
                                rel_chunk,
                                result.score * self.discount_factor,
                                len(expanded_results) + 1,
                                f"joined:reverse:{result.source}",
                            )
                        )
                        added_for_this += 1

        return [
            SearchResult(r.chunk, r.score, rank, r.source)
            for rank, r in enumerate(expanded_results, start=1)
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
