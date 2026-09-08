"""Compare in-memory and memory-mapped BM25 on the same labelled fixture."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Sequence

from .benchmark import load_dataset, run_benchmark
from .hardware import profile_hardware
from .retrievers.bm25 import BM25Retriever
from .retrievers.mmap_bm25 import MMapBM25Retriever


def compare_backends(
    dataset: str | Path,
    *,
    index_path: str | Path,
    top_k: int = 3,
    repetitions: int = 100,
) -> dict[str, object]:
    name, chunks, cases = load_dataset(dataset)
    memory = BM25Retriever()
    memory_started = perf_counter()
    memory.add(chunks)
    memory_build_ms = (perf_counter() - memory_started) * 1000
    memory_report = run_benchmark(
        memory, cases, corpus_size=len(chunks), top_k=top_k, repetitions=repetitions
    )

    disk_started = perf_counter()
    disk = MMapBM25Retriever.build(index_path, chunks)
    disk_build_ms = (perf_counter() - disk_started) * 1000
    try:
        disk_report = run_benchmark(
            disk, cases, corpus_size=len(chunks), top_k=top_k, repetitions=repetitions
        )
    finally:
        disk.close()
    index_size = sum(path.stat().st_size for path in Path(index_path).iterdir() if path.is_file())
    return {
        "dataset": name,
        "hardware": profile_hardware().to_dict(),
        "in_memory": {"build_ms": memory_build_ms, **memory_report.to_dict()},
        "memory_mapped": {
            "build_ms": disk_build_ms,
            "index_bytes": index_size,
            **disk_report.to_dict(),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare adaptive-rag BM25 backends")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.index_path:
        payload = compare_backends(
            args.dataset,
            index_path=args.index_path,
            top_k=args.top_k,
            repetitions=args.repetitions,
        )
    else:
        with tempfile.TemporaryDirectory() as directory:
            payload = compare_backends(
                args.dataset,
                index_path=Path(directory) / "benchmark-index",
                top_k=args.top_k,
                repetitions=args.repetitions,
            )
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

