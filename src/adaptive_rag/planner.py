"""Hardware-aware result budgeting and transparent confidence gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .hardware import HardwareProfile, profile_hardware
from .models import Query, RetrievalStats, SearchResult
from .retrievers.base import Retriever
from .tokenization import NormalizedTokenizer


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    capacity_tier: str
    requested_top_k: int
    effective_top_k: int
    confidence_threshold: float


@dataclass(frozen=True, slots=True)
class RetrievalDecision:
    plan: RetrievalPlan
    confidence: float
    accepted: bool
    reason: str
    raw_result_count: int


class AdaptiveRetriever:
    """Apply capacity budgets and confidence gating to a retrieval strategy."""

    def __init__(
        self,
        retriever: Retriever,
        *,
        hardware: HardwareProfile | None = None,
        confidence_threshold: float = 0.30,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.retriever = retriever
        self.hardware = hardware or profile_hardware()
        self.confidence_threshold = confidence_threshold
        self.tokenizer = tokenizer or NormalizedTokenizer()
        self.last_decision: RetrievalDecision | None = None
        self.last_stats: RetrievalStats | None = None

    def plan(self, *, top_k: int) -> RetrievalPlan:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        memory = self.hardware.physical_memory_bytes
        if memory is not None and memory < 4 * 1024**3:
            tier, limit = "constrained", 10
        elif self.hardware.logical_cpus <= 4 or (memory is not None and memory < 8 * 1024**3):
            tier, limit = "balanced", 25
        else:
            tier, limit = "high-capacity", 100
        return RetrievalPlan(tier, top_k, min(top_k, limit), self.confidence_threshold)

    def search(self, query: str | Query, *, top_k: int = 5) -> list[SearchResult]:
        plan = self.plan(top_k=top_k)
        raw = list(self.retriever.search(query, top_k=plan.effective_top_k))
        text = query.text if isinstance(query, Query) else query
        confidence = self._confidence(text, raw)
        accepted = bool(raw) and confidence >= plan.confidence_threshold
        if not raw:
            reason = "no lexical candidates"
        elif accepted:
            reason = "confidence threshold met"
        else:
            reason = "confidence below threshold"
        self.last_decision = RetrievalDecision(plan, confidence, accepted, reason, len(raw))
        underlying = getattr(self.retriever, "last_stats", None)
        if isinstance(underlying, RetrievalStats):
            self.last_stats = RetrievalStats(
                underlying.query_terms,
                underlying.candidates,
                len(raw) if accepted else 0,
                underlying.latency_ms,
                "adaptive:bm25" if underlying.strategy == "bm25" else f"adaptive:{underlying.strategy}",
            )
        return raw if accepted else []

    def _confidence(self, query: str, results: list[SearchResult]) -> float:
        if not results:
            return 0.0
        query_terms = set(self.tokenizer(query))
        if not query_terms:
            return 0.0
        result_terms = set(self.tokenizer(results[0].chunk.text))
        coverage = len(query_terms & result_terms) / len(query_terms)
        top_score = results[0].score
        second_score = results[1].score if len(results) > 1 else 0.0
        separation = max(0.0, min(1.0, (top_score - second_score) / top_score)) if top_score else 0.0
        return min(1.0, 0.75 * coverage + 0.25 * separation)

