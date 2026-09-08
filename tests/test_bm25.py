import unittest

from adaptive_rag import BM25Retriever, Chunk, Query, TelemetryCollector


class BM25Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = TelemetryCollector()
        self.retriever = BM25Retriever(telemetry=self.telemetry)
        self.retriever.add(
            [
                Chunk("a", "one", "red apple fresh fruit"),
                Chunk("b", "two", "blue ocean deep water"),
                Chunk("c", "three", "apple pie recipe apple"),
            ]
        )

    def test_ranks_relevant_result_first(self) -> None:
        results = self.retriever.search(Query("deep ocean"), top_k=2)
        self.assertEqual(results[0].chunk.id, "b")
        self.assertEqual(results[0].rank, 1)
        self.assertEqual(results[0].source, "bm25")

    def test_term_frequency_affects_score(self) -> None:
        results = self.retriever.search("apple", top_k=2)
        self.assertEqual([result.chunk.id for result in results], ["c", "a"])

    def test_unknown_terms_return_no_results(self) -> None:
        self.assertEqual(self.retriever.search("volcano"), [])

    def test_duplicate_ids_are_rejected_atomically(self) -> None:
        with self.assertRaises(ValueError):
            self.retriever.add([Chunk("new", "d", "x"), Chunk("new", "d", "y")])
        self.assertEqual(len(self.retriever), 3)

    def test_stats_and_telemetry_are_recorded(self) -> None:
        self.retriever.search("apple")
        self.assertIsNotNone(self.retriever.last_stats)
        names = [metric.name for metric in self.telemetry.snapshot()]
        self.assertEqual(names, ["bm25.index", "bm25.search"])


if __name__ == "__main__":
    unittest.main()

