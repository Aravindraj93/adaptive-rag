import unittest

from adaptive_rag.ann_benchmark import run_ann_benchmark


class AnnBenchmarkTests(unittest.TestCase):
    def test_small_benchmark_reports_candidate_reduction(self) -> None:
        report = run_ann_benchmark(documents=100, queries=10, dimensions=32)
        self.assertEqual(report["exact"]["top1_accuracy"], 1.0)
        self.assertGreaterEqual(report["lsh"]["recall_at_1_vs_exact"], 0.8)
        self.assertLess(report["lsh"]["candidate_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
