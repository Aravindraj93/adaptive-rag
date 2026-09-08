import unittest
from evaluate_phase10 import choose


def observation(features, primary_quality, fusion_quality=1):
    return dict(features=features, primary=dict(ids=['a'], ms=1, recall=primary_quality,
                                               ndcg=primary_quality),
                fusion=dict(ms=10, recall=fusion_quality, ndcg=fusion_quality), reuse_extra_ms=9)


class Phase10ProtocolTests(unittest.TestCase):
    def test_all_primary_when_equally_good(self):
        chosen = choose([observation([1,0,0], 1)])
        self.assertEqual(chosen['selected']['acceptance'], 1)
        self.assertEqual(chosen['selected']['ms'], 1)
        self.assertEqual(len(chosen['candidates']), 316)

    def test_separates_useful_feature(self):
        result = choose([observation([0,0,1], 1), observation([0,0,0], 0)])['selected']
        self.assertEqual(result['weights'], [0,0,1])
        self.assertEqual(result['acceptance'], .5)
        self.assertEqual(result['recall'], 1)

    def test_always_fusion_wins_tie_without_gate(self):
        result = choose([observation([0,0,0], 0)])['selected']
        self.assertIsNone(result['threshold'])

    def test_empty_primary_cannot_be_accepted(self):
        row = observation([1,1,1], 0)
        row['primary']['ids'] = []
        result = choose([row])['selected']
        self.assertIsNone(result['threshold'])

    def test_empty_development_rejected(self):
        with self.assertRaises(ValueError):
            choose([])


if __name__ == '__main__':
    unittest.main()
