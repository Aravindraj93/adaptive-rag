"""Opt-in, stateless candidate reuse with transparent generic confidence features."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from .composition import ReciprocalRankFusionRetriever
from .models import Query, SearchResult
from .tokenization import NormalizedTokenizer


def confidence_features(text: str, results: Sequence[SearchResult],
                        tokenizer: Callable[[str], list[str]]) -> tuple[float, float, float]:
    """Top-result term coverage, relative score gap, and top-10 concentration.

    Features are bounded heuristics, not probabilities. Scores are interpreted
    as nonnegative relevance scores; invalid or negative scores contribute zero.
    """
    terms = set(tokenizer(text))
    if not results or not terms:
        return (0.0, 0.0, 0.0)
    coverage = len(terms & set(tokenizer(results[0].chunk.text))) / len(terms)
    scores = [max(0.0, r.score) if math.isfinite(r.score) else 0.0 for r in results[:10]]
    top = scores[0]
    gap = max(0.0, min(1.0, (top-scores[1])/top)) if len(scores) > 1 and top else float(top > 0)
    # Scale by the maximum to prevent overflow with large finite scores.
    scale = max(scores)
    total = sum(s/scale for s in scores) if scale else 0.0
    concentration = ((len(scores)*(top/scale)/total-1)/(len(scores)-1)
                     if total and len(scores) > 1 else float(top > 0))
    return coverage, gap, max(0.0, min(1.0, concentration))


@dataclass(frozen=True, slots=True)
class FusionPolicy:
    """Explicit opt-in policy. None threshold selects ordinary always-fusion.

    Weights must sum to one. Calibrate on representative development data;
    there is deliberately no universal confidence threshold default.
    """
    threshold: float | None = None
    weights: tuple[float, float, float] = (0.75, 0.25, 0.0)

    def __post_init__(self) -> None:
        if self.threshold is not None and (not math.isfinite(self.threshold)
                                          or not 0 <= self.threshold <= 1):
            raise ValueError('threshold must be None or a finite value in [0, 1]')
        weights = tuple(self.weights)
        if (len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights)
                or not math.isclose(sum(weights), 1.0, abs_tol=1e-9)):
            raise ValueError('three finite nonnegative weights must sum to one')
        object.__setattr__(self, 'weights', weights)

    def confidence(self, features: Sequence[float]) -> float:
        if len(features) != 3 or any(not math.isfinite(f) or not 0 <= f <= 1 for f in features):
            raise ValueError('three bounded finite features required')
        return min(1.0, sum(w*f for w, f in zip(self.weights, features)))


@dataclass(frozen=True, slots=True)
class FusionDecision:
    route: str
    confidence: float | None
    features: tuple[float, float, float] | None
    candidate_count: int
    reused_primary: bool


class AdaptiveFusionRetriever:
    """Gate the first fusion backend and reuse its exact ranked candidates.

    The first backend should be a nonnegative lexical scorer. It is queried at
    fusion depth, not output depth, to preserve ordinary RRF fallback rankings.
    Query objects are passed through unchanged, including metadata filters.
    Prefetched candidates are local to one call, never retained for another query.
    Like other wrappers, last_decision is diagnostic and not thread-local.
    No hardware top-k truncation is applied: callers control candidate budgets.
    """
    def __init__(self, fusion: ReciprocalRankFusionRetriever, *, policy: FusionPolicy,
                 tokenizer: Callable[[str], list[str]] | None = None) -> None:
        self.fusion = fusion
        self.policy = policy
        self.tokenizer = tokenizer or NormalizedTokenizer()
        self.last_decision: FusionDecision | None = None

    def search(self, query: str | Query, *, top_k: int = 5) -> list[SearchResult]:
        self.last_decision = None
        if top_k < 1:
            raise ValueError('top_k must be at least one')
        if self.policy.threshold is None:
            results = self.fusion.search(query, top_k=top_k)
            self.last_decision = FusionDecision('fusion', None, None, 0, False)
            return results
        primary = list(self.fusion.retrievers[0].search(
            query, top_k=top_k*self.fusion.candidate_multiplier))
        features = confidence_features(query.text if isinstance(query, Query) else query,
                                       primary, self.tokenizer)
        confidence = self.policy.confidence(features)
        if primary and confidence >= self.policy.threshold:
            self.last_decision = FusionDecision('primary', confidence, features, len(primary), False)
            return primary[:top_k]
        results = self.fusion._search_with_prefetched(query, top_k=top_k, primary_results=primary)
        self.last_decision = FusionDecision('fusion', confidence, features, len(primary), True)
        return results
