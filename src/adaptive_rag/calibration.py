"""Confidence-threshold calibration against labelled retrieval cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .benchmark import BenchmarkCase
from .hardware import HardwareProfile
from .planner import AdaptiveRetriever
from .retrievers.base import Retriever


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    threshold: float
    precision: float
    recall: float
    f1: float
    acceptance_rate: float
    observations: int


def calibrate_confidence(
    retriever: Retriever,
    cases: Iterable[BenchmarkCase],
    *,
    hardware: HardwareProfile | None = None,
    top_k: int = 3,
    thresholds: Sequence[float] | None = None,
) -> CalibrationReport:
    """Choose the threshold with the best F1 for accepting correct retrievals."""

    materialized = list(cases)
    if not materialized:
        raise ValueError("at least one calibration case is required")
    candidates = tuple(thresholds or [value / 20 for value in range(21)])
    if not candidates or any(not 0 <= value <= 1 for value in candidates):
        raise ValueError("thresholds must contain values between 0 and 1")

    observations: list[tuple[float, bool]] = []
    adaptive = AdaptiveRetriever(retriever, hardware=hardware, confidence_threshold=0.0)
    for case in materialized:
        results = adaptive.search(case.query, top_k=top_k)
        decision = adaptive.last_decision
        assert decision is not None
        correct = bool({result.chunk.id for result in results} & case.relevant_chunk_ids)
        observations.append((decision.confidence, correct))

    total_correct = sum(correct for _, correct in observations)
    reports: list[CalibrationReport] = []
    for threshold in candidates:
        accepted = [(confidence, correct) for confidence, correct in observations if confidence >= threshold]
        true_positive = sum(correct for _, correct in accepted)
        precision = true_positive / len(accepted) if accepted else 0.0
        recall = true_positive / total_correct if total_correct else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        reports.append(
            CalibrationReport(
                threshold,
                precision,
                recall,
                f1,
                len(accepted) / len(observations),
                len(observations),
            )
        )
    return max(reports, key=lambda report: (report.f1, report.precision, report.threshold))

