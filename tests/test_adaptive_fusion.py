import math
import unittest
from adaptive_rag import (AdaptiveFusionRetriever, FusionPolicy, BM25Retriever,
                          Chunk, Query, SearchResult, ReciprocalRankFusionRetriever)
from adaptive_rag.adaptive_fusion import confidence_features


class Spy:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def search(self, query, *, top_k=5):
        self.calls.append((query, top_k))
        return self.rows[:top_k]


def rows(prefix='d', count=30):
    return [SearchResult(Chunk(prefix+str(i), prefix+str(i), 'alpha beta'),
                         1.0, i+1, prefix) for i in range(count)]


class AdaptiveFusionTests(unittest.TestCase):
    def test_fallback_reuses_full_depth_and_matches_fusion(self):
        primary, secondary = Spy(rows()), Spy(list(reversed(rows('x'))))
        fusion = ReciprocalRankFusionRetriever([primary, secondary])
        expected = fusion.search('unmatched', top_k=10)
        primary.calls.clear()
        secondary.calls.clear()
        router = AdaptiveFusionRetriever(fusion, policy=FusionPolicy(.9))
        self.assertEqual(router.search('unmatched', top_k=10), expected)
        self.assertEqual(len(primary.calls), 1)
        self.assertEqual(primary.calls[0][1], 30)
        self.assertEqual(len(secondary.calls), 1)
        self.assertTrue(router.last_decision.reused_primary)

    def test_acceptance_avoids_secondary_and_truncates(self):
        a, b = Spy(rows()), Spy(rows('x'))
        router = AdaptiveFusionRetriever(ReciprocalRankFusionRetriever([a,b]),
                                         policy=FusionPolicy(0))
        self.assertEqual(router.search('alpha', top_k=3), rows()[:3])
        self.assertEqual(b.calls, [])

    def test_empty_primary_is_reused_not_refetched(self):
        a, b = Spy([]), Spy(rows('x'))
        router = AdaptiveFusionRetriever(ReciprocalRankFusionRetriever([a,b]),
                                         policy=FusionPolicy(0))
        self.assertTrue(router.search('alpha'))
        self.assertEqual(len(a.calls), 1)
        self.assertTrue(router.last_decision.reused_primary)

    def test_query_object_passed_unchanged(self):
        a, b = Spy([]), Spy([])
        q = Query('alpha', {'filter': {'tenant': 'a'}})
        AdaptiveFusionRetriever(ReciprocalRankFusionRetriever([a,b]),
                                policy=FusionPolicy(1)).search(q)
        self.assertIs(a.calls[0][0], q)
        self.assertIs(b.calls[0][0], q)

    def test_no_cross_query_reuse(self):
        a, b = Spy([]), Spy([])
        router = AdaptiveFusionRetriever(ReciprocalRankFusionRetriever([a,b]),
                                         policy=FusionPolicy(1))
        router.search('first')
        router.search('second')
        self.assertEqual([q for q, _ in a.calls], ['first', 'second'])

    def test_always_fusion_matches_unwrapped(self):
        a, b = Spy(rows()), Spy(rows('x'))
        fusion = ReciprocalRankFusionRetriever([a,b])
        router = AdaptiveFusionRetriever(fusion, policy=FusionPolicy())
        self.assertEqual(router.search('alpha'), fusion.search('alpha'))
        self.assertIsNone(router.last_decision.confidence)

    def test_weight_validation_and_immutability(self):
        for weights in [(1, 1, 1), (math.nan, 0, 1), (-1, 1, 1), (1, 0)]:
            with self.assertRaises(ValueError):
                FusionPolicy(weights=weights)
        values = [1, 0, 0]
        policy = FusionPolicy(weights=values)
        values[0] = 0
        self.assertEqual(policy.weights, (1, 0, 0))

    def test_threshold_and_feature_validation(self):
        for threshold in [-1, 2, math.nan, math.inf]:
            with self.assertRaises(ValueError):
                FusionPolicy(threshold)
        with self.assertRaises(ValueError):
            FusionPolicy().confidence([1, math.nan, 0])

    def test_features_ties_empty_and_extreme_scores(self):
        self.assertEqual(confidence_features('alpha', [], str.split), (0, 0, 0))
        self.assertEqual(confidence_features('alpha', rows(), str.split), (1, 0, 0))
        extreme = [SearchResult(r.chunk, 1e308, r.rank, r.source) for r in rows()]
        self.assertTrue(all(math.isfinite(f) for f in confidence_features('alpha', extreme, str.split)))

    def test_threshold_equality_and_invalid_topk(self):
        a, b = Spy(rows()), Spy([])
        router = AdaptiveFusionRetriever(ReciprocalRankFusionRetriever([a,b]),
                                         policy=FusionPolicy(1, (1, 0, 0)))
        self.assertTrue(router.search('alpha'))
        self.assertEqual(router.last_decision.route, 'primary')
        with self.assertRaises(ValueError):
            router.search('alpha', top_k=0)
        self.assertIsNone(router.last_decision)


if __name__ == '__main__':
    unittest.main()
