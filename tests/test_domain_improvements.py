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
        tokens_alert = tokenizer("ALERT_SYS101")
        self.assertIn("alert", tokens_alert)
        self.assertIn("sys101", tokens_alert)
        self.assertIn("sys", tokens_alert)
        self.assertIn("101", tokens_alert)

        tokens_req = tokenizer("REQ_SYS081: SystemActivation requirement")
        self.assertIn("req", tokens_req)
        self.assertIn("sys081", tokens_req)
        self.assertIn("sys", tokens_req)
        self.assertIn("081", tokens_req)
        self.assertIn("system", tokens_req)
        self.assertIn("activation", tokens_req)

        tokens_camel = tokenizer("SystemActivation")
        self.assertIn("system", tokens_camel)
        self.assertIn("activation", tokens_camel)

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
            Chunk("req-1", "doc-1", "REQ_SYS081: SystemActivation selection requirement."),
            Chunk("req-2", "doc-1", "General overview and system explanation."),
        ])

        embedder = HashingEmbedder(dimensions=16)
        dense = DenseRetriever(embedder, min_score=None)
        dense.add([
            Chunk("req-1", "doc-1", "REQ_SYS081: SystemActivation selection requirement."),
            Chunk("req-2", "doc-1", "General overview and system explanation."),
        ])

        fusion_vanilla = ReciprocalRankFusionRetriever([bm25, dense], weights=[1.0, 1.0])
        results_vanilla = fusion_vanilla.search("REQ_SYS081", top_k=2)
        self.assertEqual(len(results_vanilla), 2)

        fusion_boosted = ReciprocalRankFusionRetriever(
            [bm25, dense], weights=[1.0, 1.0], anchor_boost=1.0, anchor_gap_threshold=0.1
        )
        results_boosted = fusion_boosted.search("REQ_SYS081", top_k=2)
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
            "dialog-101": Chunk(
                "dialog-101", "doc-p", "SYS101: System Dialog window.",
                metadata={"type": "dialog", "trigger_id": "trigger-101"}
            ),
            "trigger-101": Chunk(
                "trigger-101", "doc-p", "Activation button pressed triggers SYS101.",
                metadata={"type": "trigger", "parent_id": "dialog-101"}
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
        results = rel_retriever.search("SYS101", top_k=1)
        result_ids = [r.chunk.id for r in results]
        self.assertIn("dialog-101", result_ids)
        self.assertIn("trigger-101", result_ids)
        self.assertNotIn("other-chunk", result_ids)

    def test_metadata_score_modifier(self):
        bm25 = BM25Retriever()
        bm25.add([
            Chunk("hist-1", "d1", "System change log revision notes.", metadata={"source_type": "history"}),
            Chunk("req-1", "d1", "System specification active requirement.", metadata={"source_type": "requirement"}),
        ])
        modifier = MetadataScoreModifier(
            bm25,
            multipliers={"source_type": {"requirement": 2.0, "history": 0.2}},
        )
        results = modifier.search("System", top_k=2)
        self.assertEqual(results[0].chunk.id, "req-1")
        self.assertEqual(results[1].chunk.id, "hist-1")

    def test_fastpath_id_lookup(self):
        chunks = {
            "SYS081": Chunk("req-81", "d1", "SYS081: Full Requirement Description"),
            "SYS101": Chunk("dialog-101", "d1", "SYS101: Dialog Description"),
        }
        fallback = BM25Retriever()
        fallback.add([
            Chunk("other-1", "d1", "General system overview without ID."),
        ])
        fastpath = FastPathIDLookupRetriever(fallback, chunks)

        results_req = fastpath.search("REQ_SYS081", top_k=2)
        self.assertEqual(results_req[0].chunk.id, "req-81")
        self.assertEqual(results_req[0].source, "fastpath:exact_id")

        results_alert = fastpath.search("ALERT_SYS101", top_k=2)
        self.assertEqual(results_alert[0].chunk.id, "dialog-101")

        results_fb = fastpath.search("overview", top_k=1)
        self.assertEqual(results_fb[0].chunk.id, "other-1")
        self.assertEqual(results_fb[0].source, "bm25")


if __name__ == "__main__":
    unittest.main()
