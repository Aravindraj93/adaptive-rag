"""Reproducible retrieval benchmark harness and JSON fixture support."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

from .hardware import profile_hardware
from .models import Chunk, Query
from .retrievers.base import Retriever
from .retrievers.bm25 import BM25Retriever


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    query: Query
    relevant_chunk_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    corpus_size: int
    query_count: int
    top_k: int
    recall_at_k: float
    mean_reciprocal_rank: float
    mean_latency_ms: float
    p95_latency_ms: float
    throughput_queries_per_second: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def run_benchmark(
    retriever: Retriever,
    cases: Iterable[BenchmarkCase],
    *,
    corpus_size: int,
    top_k: int = 5,
    repetitions: int = 1,
) -> BenchmarkReport:
    """Evaluate recall, reciprocal rank, latency, and throughput."""

    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    materialized = list(cases)
    if not materialized:
        raise ValueError("at least one benchmark case is required")

    recall_scores: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    for _ in range(repetitions):
        for case in materialized:
            started = perf_counter()
            results = retriever.search(case.query, top_k=top_k)
            latencies.append((perf_counter() - started) * 1000)
            ranked_ids = [result.chunk.id for result in results]
            relevant_ranks = [
                rank
                for rank, chunk_id in enumerate(ranked_ids, start=1)
                if chunk_id in case.relevant_chunk_ids
            ]
            retrieved_relevant = len(set(ranked_ids) & case.relevant_chunk_ids)
            recall_scores.append(
                retrieved_relevant / len(case.relevant_chunk_ids)
                if case.relevant_chunk_ids
                else 0.0
            )
            reciprocal_ranks.append(1 / min(relevant_ranks) if relevant_ranks else 0.0)

    executions = len(materialized) * repetitions
    sorted_latency = sorted(latencies)
    p95_index = max(0, math_ceil(0.95 * len(sorted_latency)) - 1)
    total_seconds = sum(latencies) / 1000
    return BenchmarkReport(
        corpus_size=corpus_size,
        query_count=executions,
        top_k=top_k,
        recall_at_k=statistics.fmean(recall_scores),
        mean_reciprocal_rank=statistics.fmean(reciprocal_ranks),
        mean_latency_ms=statistics.fmean(latencies),
        p95_latency_ms=sorted_latency[p95_index],
        throughput_queries_per_second=executions / total_seconds if total_seconds else float("inf"),
    )


def math_ceil(value: float) -> int:
    integer = int(value)
    return integer if integer == value else integer + 1


def sample_dataset() -> tuple[list[Chunk], list[BenchmarkCase]]:
    chunks = [
        Chunk("doc-1", "guide", "Reset an account password from the security settings page."),
        Chunk("doc-2", "policy", "Refund requests are accepted within thirty calendar days."),
        Chunk("doc-3", "manual", "Solar panels convert sunlight into electrical energy."),
        Chunk("doc-4", "handbook", "New employees receive equipment during onboarding."),
        Chunk("doc-5", "recipe", "Bread dough rises when yeast produces carbon dioxide."),
        Chunk("doc-6", "travel", "The express train reaches the airport in twenty minutes."),
    ]
    cases = [
        BenchmarkCase(Query("How can I reset my password?"), frozenset({"doc-1"})),
        BenchmarkCase(Query("What is the refund period?"), frozenset({"doc-2"})),
        BenchmarkCase(Query("How do solar panels make electricity?"), frozenset({"doc-3"})),
        BenchmarkCase(Query("When do staff get their equipment?"), frozenset({"doc-4"})),
        BenchmarkCase(Query("What makes bread dough rise?"), frozenset({"doc-5"})),
        BenchmarkCase(Query("How long is the airport express train?"), frozenset({"doc-6"})),
    ]
    return chunks, cases


def load_dataset(path: str | Path) -> tuple[str, list[Chunk], list[BenchmarkCase]]:
    """Load a JSON benchmark fixture with generic chunks and relevance labels."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        name = str(payload["name"])
        chunks = [
            Chunk(item["id"], item["document_id"], item["text"], item.get("metadata", {}))
            for item in payload["chunks"]
        ]
        cases = [
            BenchmarkCase(Query(item["query"]), frozenset(item["relevant_chunk_ids"]))
            for item in payload["cases"]
        ]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"invalid benchmark dataset: {error}") from error
    if not name or not chunks or not cases:
        raise ValueError("benchmark dataset name, chunks, and cases must not be empty")
    chunk_ids = {chunk.id for chunk in chunks}
    if len(chunk_ids) != len(chunks):
        raise ValueError("benchmark chunk ids must be unique")
    if any(not case.relevant_chunk_ids <= chunk_ids for case in cases):
        raise ValueError("benchmark case references an unknown chunk id")
    return name, chunks, cases


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an adaptive-rag BM25 benchmark")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", type=Path, help="JSON fixture; defaults to smoke data")
    args = parser.parse_args(argv)

    if args.dataset:
        dataset_name, chunks, cases = load_dataset(args.dataset)
    else:
        dataset_name = "smoke-v1"
        chunks, cases = sample_dataset()
    retriever = BM25Retriever()
    retriever.add(chunks)
    report = run_benchmark(
        retriever,
        cases,
        corpus_size=len(chunks),
        top_k=args.top_k,
        repetitions=args.repetitions,
    )
    payload = {
        "dataset": dataset_name,
        "hardware": profile_hardware().to_dict(),
        "benchmark": report.to_dict(),
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())