"""Synthetic scale benchmark for memory, startup, size, quality, and latency."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import tracemalloc
from pathlib import Path
from time import perf_counter
from typing import Iterator, Sequence

from .models import Chunk
from .retrievers.bm25 import BM25Retriever
from .retrievers.mmap_bm25 import MMapBM25Retriever


def synthetic_chunks(count: int) -> Iterator[Chunk]:
    for number in range(count):
        topic = f"topic{number:07d}"
        yield Chunk(
            topic,
            f"document{number:07d}",
            f"Reference entry {topic} contains marker{number:07d} and general archive text.",
            {"partition": number % 10},
        )


def _measure_build(factory):
    tracemalloc.start()
    started = perf_counter()
    value = factory()
    elapsed_ms = (perf_counter() - started) * 1000
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, elapsed_ms, retained, peak


def _measure_queries(retriever, document_count: int, query_count: int) -> dict[str, float]:
    latencies = []
    correct = 0
    for query_number in range(query_count):
        target = (query_number * 104729) % document_count
        query = f"topic{target:07d} marker{target:07d}"
        started = perf_counter()
        results = retriever.search(query, top_k=1)
        latencies.append((perf_counter() - started) * 1000)
        correct += bool(results and results[0].chunk.id == f"topic{target:07d}")
    return {
        "top1_accuracy": correct / query_count,
        "mean_latency_ms": statistics.fmean(latencies),
        "p95_latency_ms": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
    }


def run_scale_benchmark(
    *,
    documents: int = 10_000,
    queries: int = 100,
    index_path: str | Path,
) -> dict[str, object]:
    if documents < 1 or queries < 1:
        raise ValueError("documents and queries must be positive")
    memory, memory_build_ms, memory_retained, memory_peak = _measure_build(
        lambda: _build_memory(documents)
    )
    memory_queries = _measure_queries(memory, documents, queries)

    disk, disk_build_ms, disk_retained, disk_peak = _measure_build(
        lambda: MMapBM25Retriever.build(index_path, synthetic_chunks(documents))
    )
    disk.close()
    startup_started = perf_counter()
    disk = MMapBM25Retriever(index_path)
    startup_ms = (perf_counter() - startup_started) * 1000
    try:
        disk_queries = _measure_queries(disk, documents, queries)
    finally:
        disk.close()
    index_bytes = sum(path.stat().st_size for path in Path(index_path).iterdir() if path.is_file())
    return {
        "documents": documents,
        "queries": queries,
        "in_memory": {
            "build_ms": memory_build_ms,
            "retained_python_bytes": memory_retained,
            "peak_python_bytes": memory_peak,
            **memory_queries,
        },
        "memory_mapped": {
            "build_ms": disk_build_ms,
            "retained_python_bytes": disk_retained,
            "peak_python_bytes": disk_peak,
            "startup_ms": startup_ms,
            "index_bytes": index_bytes,
            **disk_queries,
        },
    }


def _build_memory(documents: int) -> BM25Retriever:
    retriever = BM25Retriever()
    retriever.add(synthetic_chunks(documents))
    return retriever


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run adaptive-rag synthetic scale benchmark")
    parser.add_argument("--documents", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.index_path:
        payload = run_scale_benchmark(
            documents=args.documents, queries=args.queries, index_path=args.index_path
        )
    else:
        with tempfile.TemporaryDirectory() as directory:
            payload = run_scale_benchmark(
                documents=args.documents,
                queries=args.queries,
                index_path=Path(directory) / "scale-index",
            )
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

