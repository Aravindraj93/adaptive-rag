import unittest

from adaptive_rag import BM25Retriever, Chunk
from adaptive_rag.composition import ReciprocalRankFusionRetriever, SelectiveRerankingRetriever
from adaptive_rag.retrievers.dense import DenseRetriever


def keyword_embedder(texts):
    vocabulary = ("apple", "ocean", "policy")
    return [[float(word in text.casefold()) for word in vocabulary] for text in texts]


class DenseTests(unittest.TestCase):
    def test_application_supplied_embeddings_support_cosine_search(self) -> None:
        retriever = DenseRetriever(keyword_embedder)
        retriever.add([
            Chunk("a", "d", "apple orchard"),
            Chunk("b", "d", "deep ocean"),
        ])
        self.assertEqual(retriever.search("ocean")[0].chunk.id, "b")
        self.assertEqual(retriever.search("policy"), [])

    def test_embedding_dimension_mismatch_is_rejected(self) -> None:
        state = {"query": False}

        def inconsistent(texts):
            if state["query"]:
                return [[1.0, 0.0, 0.0]]
            state["query"] = True
            return [[1.0, 0.0] for _ in texts]

        retriever = DenseRetriever(inconsistent)
        retriever.add([Chunk("a", "d", "text")])
        with self.assertRaisesRegex(ValueError, "dimensions"):
            retriever.search("query")


class CompositionTests(unittest.TestCase):
    def test_rrf_combines_lexical_and_dense_rankings(self) -> None:
        chunks = [
            Chunk("a", "d", "apple orchard"),
            Chunk("b", "d", "apple ocean"),
        ]
        lexical = BM25Retriever()
        lexical.add(chunks)
        dense = DenseRetriever(keyword_embedder)
        dense.add(chunks)
        fused = ReciprocalRankFusionRetriever([lexical, dense], rank_constant=10)
        results = fused.search("apple ocean", top_k=2)
        self.assertEqual(results[0].chunk.id, "b")
        self.assertTrue(results[0].source.startswith("rrf:"))

    def test_reranker_runs_only_for_ambiguous_candidates(self) -> None:
        baseline = BM25Retriever()
        baseline.add([
            Chunk("a", "d", "shared term alpha"),
            Chunk("b", "d", "shared term beta"),
        ])

        def reranker(_query, chunks):
            return [1.0 if chunk.id == "b" else 0.0 for chunk in chunks]

        selective = SelectiveRerankingRetriever(
            baseline, reranker, ambiguity_threshold=1.0
        )
        results = selective.search("shared term", top_k=1)
        self.assertTrue(selective.last_reranked)
        self.assertEqual(results[0].chunk.id, "b")


if __name__ == "__main__":
    unittest.main()

