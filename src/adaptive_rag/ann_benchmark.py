"""Compare exact dense search and LSH candidate search on synthetic data."""

from __future__ import annotations

import argparse
import json
import statistics
from time import perf_counter
from typing import Sequence

from .embedders import HashingEmbedder
from .models import Chunk
from .retrievers.dense import DenseRetriever
from .retrievers.lsh_dense import LSHDenseRetriever


def run_ann_benchmark(
    *,
    documents: int = 5_000,
    queries: int = 100,
    dimensions: int = 128,
) -> dict[str, object]:
    if documents < 1 or queries < 1:
        raise ValueError("documents and queries must be positive")
    chunks = [
        Chunk(
            f"topic{number:07d}",
            "synthetic",
            f"reference topic{number:07d} marker{number:07d} archive record",
        )
        for number in range(documents)
    ]
    embedder = HashingEmbedder(dimensions=dimensions)
    exact = DenseRetriever(embedder)
    lsh = LSHDenseRetriever(
        embedder,
        dimensions=dimensions,
        tables=8,
        bits=10,
        probe_radius=1,
        seed=17,
    )
    exact.add(chunks)
    lsh.add(chunks)
    exact_latencies: list[float] = []
    lsh_latencies: list[float] = []
    candidate_counts: list[int] = []
    recalled = 0
    correct = 0
    for query_number in range(queries):
        target = (query_number * 104729) % documents
        query = f"reference topic{target:07d} marker{target:07d}"
        started = perf_counter()
        exact_result = exact.search(query, top_k=1)
        exact_latencies.append((perf_counter() - started) * 1000)
        started = perf_counter()
        lsh_result = lsh.search(query, top_k=1)
        lsh_latencies.append((perf_counter() - started) * 1000)
        candidate_counts.append(lsh.last_stats.candidates)
        expected = f"topic{target:07d}"
        correct += bool(exact_result and exact_result[0].chunk.id == expected)
        recalled += bool(
            exact_result
            and lsh_result
            and exact_result[0].chunk.id == lsh_result[0].chunk.id
        )
    return {
        "documents": documents,
        "queries": queries,
        "dimensions": dimensions,
        "exact": {
            "top1_accuracy": correct / queries,
            "mean_latency_ms": statistics.fmean(exact_latencies),
        },
        "lsh": {
            "recall_at_1_vs_exact": recalled / queries,
            "mean_latency_ms": statistics.fmean(lsh_latencies),
            "mean_candidates": statistics.fmean(candidate_counts),
            "candidate_fraction": statistics.fmean(candidate_counts) / documents,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare exact and LSH dense retrieval")
    parser.add_argument("--documents", type=int, default=5_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    payload = run_ann_benchmark(
        documents=args.documents,
        queries=args.queries,
        dimensions=args.dimensions,
    )
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

