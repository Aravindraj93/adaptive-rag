"""Evaluation-only tests; require the evaluator's optional dependencies."""
import unittest
from evaluate_routing import metrics, select_policy


def row(confidence, primary_quality=0, fusion_quality=1, ids=None):
    return dict(confidence=confidence,
                primary=dict(ids=['a'] if ids is None else ids, recall=primary_quality,
                             ndcg=primary_quality, ms=1),
                fusion=dict(ids=['b'], recall=fusion_quality, ndcg=fusion_quality, ms=10))


class RoutingProtocolTests(unittest.TestCase):
    def test_empty_development_rejected(self):
        with self.assertRaises(ValueError):
            select_policy([])

    def test_invalid_tolerance_rejected(self):
        with self.assertRaises(ValueError):
            select_policy([row(.5)], -1)

    def test_keep_primary_when_quality_equal(self):
        selected = select_policy([row(.5, 1)])['selected']
        self.assertEqual(selected['threshold'], 0)
        self.assertEqual(selected['estimated_ms'], 1)

    def test_always_fusion_when_routing_only_adds_cost(self):
        self.assertIsNone(select_policy([row(.5)])['selected']['threshold'])

    def test_route_low_confidence_and_keep_high_confidence(self):
        selected = select_policy([row(.2), row(.8, 1)])['selected']
        self.assertEqual(selected['threshold'], .25)
        self.assertEqual(selected['estimated_ms'], 6)

    def test_confidence_at_threshold_is_accepted(self):
        selected = select_policy([row(.2), row(.25, 1)])['selected']
        self.assertEqual(selected['threshold'], .25)
        self.assertEqual(selected['recall'], 1)

    def test_empty_primary_escalates_even_with_high_confidence(self):
        selected = select_policy([row(1, ids=[])])['selected']
        self.assertIsNone(selected['threshold'])

    def test_metrics_graded_and_zero_relevance(self):
        self.assertEqual(metrics(['x'], {}), dict(recall=0, ndcg=0))
        self.assertEqual(metrics(['a', 'b'], {'a': 2, 'b': 1}), dict(recall=1, ndcg=1))
        self.assertLess(metrics(['b', 'a'], {'a': 2, 'b': 1})['ndcg'], 1)


if __name__ == '__main__':
    unittest.main()
