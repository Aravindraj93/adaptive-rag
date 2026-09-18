import unittest
from adaptive_rag import (
    BM25Retriever,
    CharNGramTokenizer,
    Chunk,
    DenseRetriever,
    FastPathIDLookupRetriever,
    HashingEmbedder,
    MetadataScoreModifier,
    NormalizedTokenizer,
    Query,
    ReciprocalRankFusionRetriever,
    RelationalExpansionRetriever,
    ScoreWeightedFusionRetriever,
)


class TestDomainImprovements(unittest.TestCase):
    def test_identifier_tokenization(self):
        tokenizer = NormalizedTokenizer(split_identifiers=True)
        tokens_popup = tokenizer("POPUP_PU1436")
        self.assertIn("popup", tokens_popup)
        self.assertIn("pu1436", tokens_popup)
        self.assertIn("pu", tokens_popup)
        self.assertIn("1436", tokens_popup)

        tokens_req = tokenizer("REQ_CFTS081: DriveMode activation")
        self.assertIn("req", tokens_req)
        self.assertIn("cfts081", tokens_req)
        self.assertIn("cfts", tokens_req)
        self.assertIn("081", tokens_req)
        self.assertIn("drive", tokens_req)
        self.assertIn("mode", tokens_req)

        tokens_camel = tokenizer("DriveMode")
        self.assertIn("drive", tokens_camel)
        self.assertIn("mode", tokens_camel)

    def test_identifier_tokenizer_persistence(self):
        tokenizer = NormalizedTokenizer(split_identifiers=True, fold_accents=True)
        payload = tokenizer.to_dict()
        self.assertTrue(payload["split_identifiers"])
        self.assertTrue(payload["fold_accents"])

        restored = NormalizedTokenizer.from_dict(payload)
        self.assertTrue(restored.split_identifiers)
        self.assertTrue(restored.fold_accents)

    def test_char_ngram_tokenizer(self):
        tokenizer = CharNGramTokenizer(min_n=3, max_n=4)
        tokens = tokenizer("SIG_STAT_REQ")
        self.assertIn("sig", tokens)
        self.assertIn("stat", tokens)
        self.assertIn("req", tokens)
        self.assertIn("tat", tokens)

        payload = tokenizer.to_dict()
        restored = CharNGramTokenizer.from_dict(payload)
        self.assertEqual(restored.min_n, 3)
        self.assertEqual(restored.max_n, 4)

    def test_rrf_anchor_boost(self):
        bm25 = BM25Retriever(tokenizer=NormalizedTokenizer(split_identifiers=True))
        bm25.add([
            Chunk("req-1", "doc-1", "REQ_CFTS081: DriveMode selection requirement."),
            Chunk("req-2", "doc-1", "General overview and DriveMode explanation."),
        ])

        embedder = HashingEmbedder(dimensions=16)
        dense = DenseRetriever(embedder, min_score=None)
        dense.add([
            Chunk("req-1", "doc-1", "REQ_CFTS081: DriveMode selection requirement."),
            Chunk("req-2", "doc-1", "General overview and DriveMode explanation."),
        ])

        fusion_vanilla = ReciprocalRankFusionRetriever([bm25, dense], weights=[1.0, 1.0])
        results_vanilla = fusion_vanilla.search("REQ_CFTS081", top_k=2)
        self.assertEqual(len(results_vanilla), 2)

        fusion_boosted = ReciprocalRankFusionRetriever(
            [bm25, dense], weights=[1.0, 1.0], anchor_boost=1.0, anchor_gap_threshold=0.1
        )
        results_boosted = fusion_boosted.search("REQ_CFTS081", top_k=2)
        self.assertEqual(results_boosted[0].chunk.id, "req-1")
        self.assertGreater(results_boosted[0].score, results_boosted[1].score)

    def test_score_weighted_fusion(self):
        r1 = BM25Retriever()
        r1.add([
            Chunk("c1", "d1", "alpha beta gamma"),
            Chunk("c2", "d1", "alpha delta"),
        ])
        r2 = BM25Retriever()
        r2.add([
            Chunk("c1", "d1", "alpha beta gamma"),
            Chunk("c2", "d1", "alpha delta"),
        ])
        fusion = ScoreWeightedFusionRetriever([r1, r2], weights=[0.8, 0.2])
        results = fusion.search("gamma", top_k=2)
        self.assertEqual(results[0].chunk.id, "c1")

    def test_relational_expansion(self):
        chunks = {
            "popup-1436": Chunk(
                "popup-1436", "doc-p", "PU1436: DriveMode Selection Popup window.",
                metadata={"type": "popup", "trigger_id": "trigger-1436"}
            ),
            "trigger-1436": Chunk(
                "trigger-1436", "doc-p", "DriveMode button pressed for > 2 seconds triggers PU1436.",
                metadata={"type": "trigger", "parent_id": "popup-1436"}
            ),
            "other-chunk": Chunk(
                "other-chunk", "doc-o", "Unrelated radio setting chunk.",
                metadata={"type": "misc"}
            ),
        }
        base_bm25 = BM25Retriever(tokenizer=NormalizedTokenizer(split_identifiers=True))
        base_bm25.add(list(chunks.values()))

        rel_retriever = RelationalExpansionRetriever(
            base_bm25,
            chunks,
            relation_keys=["trigger_id", "parent_id"],
            include_reverse_references=True,
        )
        results = rel_retriever.search("PU1436", top_k=1)
        result_ids = [r.chunk.id for r in results]
        self.assertIn("popup-1436", result_ids)
        self.assertIn("trigger-1436", result_ids)
        self.assertNotIn("other-chunk", result_ids)

    def test_metadata_score_modifier(self):
        bm25 = BM25Retriever()
        bm25.add([
            Chunk("hist-1", "d1", "DriveMode change log revision notes.", metadata={"source_type": "history"}),
            Chunk("req-1", "d1", "DriveMode specification active requirement.", metadata={"source_type": "requirement"}),
        ])
        modifier = MetadataScoreModifier(
            bm25,
            multipliers={"source_type": {"requirement": 2.0, "history": 0.2}},
        )
        results = modifier.search("DriveMode", top_k=2)
        self.assertEqual(results[0].chunk.id, "req-1")
        self.assertEqual(results[1].chunk.id, "hist-1")

    def test_fastpath_id_lookup(self):
        chunks = {
            "CFTS081": Chunk("req-81", "d1", "CFTS081: Full Requirement Description"),
            "PU1436": Chunk("pu-1436", "d1", "PU1436: Popup Description"),
        }
        fallback = BM25Retriever()
        fallback.add([
            Chunk("other-1", "d1", "General system overview without ID."),
        ])
        fastpath = FastPathIDLookupRetriever(fallback, chunks)

        results_req = fastpath.search("REQ_CFTS081", top_k=2)
        self.assertEqual(results_req[0].chunk.id, "req-81")
        self.assertEqual(results_req[0].source, "fastpath:exact_id")

        results_popup = fastpath.search("POPUP_PU1436", top_k=2)
        self.assertEqual(results_popup[0].chunk.id, "pu-1436")

        results_fb = fastpath.search("overview", top_k=1)
        self.assertEqual(results_fb[0].chunk.id, "other-1")
        self.assertEqual(results_fb[0].source, "bm25")


if __name__ == "__main__":
    unittest.main()
