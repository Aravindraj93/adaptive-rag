import unittest

from adaptive_rag import BM25Retriever, Chunk, Query
from adaptive_rag.benchmark import BenchmarkCase, load_dataset, sample_dataset, run_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_sample_benchmark_produces_valid_metrics(self) -> None:
        chunks, cases = sample_dataset()
        retriever = BM25Retriever()
        retriever.add(chunks)
        report = run_benchmark(retriever, cases, corpus_size=len(chunks), top_k=3)
        self.assertEqual(report.corpus_size, 6)
        self.assertEqual(report.query_count, 6)
        self.assertGreaterEqual(report.recall_at_k, 0.8)
        self.assertGreaterEqual(report.mean_reciprocal_rank, 0.8)
        self.assertGreaterEqual(report.mean_latency_ms, 0)

    def test_recall_counts_partial_multi_relevance(self) -> None:
        chunks = [
            Chunk("a", "d", "common alpha"),
            Chunk("b", "d", "common beta"),
        ]
        retriever = BM25Retriever()
        retriever.add(chunks)
        report = run_benchmark(
            retriever,
            [BenchmarkCase(Query("alpha"), frozenset({"a", "b"}))],
            corpus_size=2,
            top_k=1,
        )
        self.assertEqual(report.recall_at_k, 0.5)

    def test_multilingual_benchmark_dataset(self) -> None:
        name, chunks, cases = load_dataset("benchmarks/multilingual_benchmark.json")
        self.assertEqual(name, "multilingual-benchmark-v1")
        self.assertGreaterEqual(len(chunks), 15)
        self.assertGreaterEqual(len(cases), 14)

        retriever = BM25Retriever()
        retriever.add(chunks)
        report = run_benchmark(retriever, cases, corpus_size=len(chunks), top_k=3)
        self.assertGreaterEqual(report.recall_at_k, 0.85)


if __name__ == "__main__":
    unittest.main()

