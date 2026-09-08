"""Confidence-driven escalation between inexpensive and richer retrievers."""

from __future__ import annotations

from dataclasses import dataclass

from .hardware import HardwareProfile
from .models import Query, SearchResult
from .planner import AdaptiveRetriever, RetrievalDecision
from .retrievers.base import Retriever


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: str
    reason: str
    primary: RetrievalDecision


class EscalatingRetriever:
    """Use the primary result when confident and otherwise query a fallback."""

    def __init__(
        self,
        primary: Retriever,
        fallback: Retriever,
        *,
        confidence_threshold: float = 0.30,
        hardware: HardwareProfile | None = None,
    ) -> None:
        self.primary = AdaptiveRetriever(
            primary,
            confidence_threshold=confidence_threshold,
            hardware=hardware,
        )
        self.fallback = fallback
        self.last_route: RouteDecision | None = None

    def search(self, query: str | Query, *, top_k: int = 5) -> list[SearchResult]:
        results = self.primary.search(query, top_k=top_k)
        decision = self.primary.last_decision
        assert decision is not None
        if results:
            self.last_route = RouteDecision("primary", decision.reason, decision)
            return results
        fallback_results = list(self.fallback.search(query, top_k=top_k))
        self.last_route = RouteDecision("fallback", decision.reason, decision)
        return [
            SearchResult(result.chunk, result.score, rank, f"fallback:{result.source}")
            for rank, result in enumerate(fallback_results, start=1)
        ]

