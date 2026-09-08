import tempfile
import unittest
from pathlib import Path

from adaptive_rag.backend_benchmark import compare_backends


class BackendBenchmarkTests(unittest.TestCase):
    def test_comparison_reports_both_backends(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = compare_backends(
                "benchmarks/general_knowledge.json",
                index_path=Path(directory) / "index",
                repetitions=1,
            )
        self.assertEqual(payload["dataset"], "general-knowledge-v1")
        self.assertEqual(payload["in_memory"]["recall_at_k"], 1.0)
        self.assertEqual(payload["memory_mapped"]["recall_at_k"], 1.0)
        self.assertGreater(payload["memory_mapped"]["index_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
