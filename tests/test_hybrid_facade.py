import unittest
from adaptive_rag import Chunk, HashingEmbedder, HybridRetriever, Query


class TestHybridFacade(unittest.TestCase):
    def setUp(self):
        self.embedder = HashingEmbedder(dimensions=16)
        self.chunks = [
            Chunk("c1", "doc1", "REQ_CFTS081: DriveMode controller specification.", {"type": "req"}),
            Chunk("c2", "doc1", "POPUP_PU1436: Interface display window.", {"type": "popup"}),
            Chunk("c3", "doc2", "General safety and battery guidelines.", {"type": "guide"}),
        ]

    def test_basic_hybrid_search(self):
        retriever = HybridRetriever(
            self.embedder,
            split_identifiers=True,
            anchor_boost=1.0,
        )
        retriever.add(self.chunks)
        self.assertEqual(len(retriever), 3)

        # Exact ID match
        results_req = retriever.search("REQ_CFTS081", top_k=2)
        self.assertGreaterEqual(len(results_req), 1)
        self.assertEqual(results_req[0].chunk.id, "c1")
        self.assertEqual(results_req[0].rank, 1)

        # Query object support
        results_query_obj = retriever.search(Query("REQ_CFTS081"), top_k=1)
        self.assertEqual(results_query_obj[0].chunk.id, "c1")

    def test_cached_hybrid_search(self):
        revision_state = ["v1"]
        retriever = HybridRetriever(
            self.embedder,
            split_identifiers=True,
            anchor_boost=1.0,
            revision=lambda: revision_state[0],
            scope="test-scope",
        )
        retriever.add(self.chunks)

        first_pass = retriever.search("POPUP_PU1436", top_k=2)
        second_pass = retriever.search("POPUP_PU1436", top_k=2)
        self.assertEqual(first_pass, second_pass)
        self.assertEqual(first_pass[0].chunk.id, "c2")


if __name__ == "__main__":
    unittest.main()
