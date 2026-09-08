import hashlib
import math
import unittest
from conservative_policy import calibrate, bootstrap_lower, accepts
from evaluate_phase11 import ExcludeSelf
from adaptive_rag import Chunk, Query, SearchResult


def rows(n=60):
    return [dict(query_id=str(i), features=[1,0,0],
                 primary=dict(ids=['doc'], recall=1, ndcg=1, ms=1),
                 fusion=dict(recall=1, ndcg=1, ms=10), reuse_extra_ms=9) for i in range(n)]


class ConservativeProtocolTests(unittest.TestCase):
    def test_equal_quality_accepts_cheaper_policy(self):
        report = calibrate(rows())
        self.assertTrue(report['approved'])
        self.assertEqual(report['selected']['threshold'], 0)

    def test_guard_rejects_one_proposal_without_retuning(self):
        data = rows()
        ordered = sorted(data, key=lambda r: hashlib.sha256(('phase11:'+r['query_id']).encode()).hexdigest())
        for row in ordered[40:]:
            row['primary'].update(recall=.5, ndcg=.5)
        report = calibrate(data)
        self.assertEqual(report['proposal']['threshold'], 0)
        self.assertFalse(report['approved'])
        self.assertIsNone(report['selected']['threshold'])
        self.assertEqual(report['guard_bounds']['ndcg']['lower'], -.5)

    def test_split_is_disjoint_deterministic_and_complete(self):
        a, b = calibrate(rows()), calibrate(list(reversed(rows())))
        self.assertEqual(a, b)
        self.assertFalse(set(a['fit_ids']) & set(a['guard_ids']))
        self.assertEqual(len(a['fit_ids'])+len(a['guard_ids']), 60)

    def test_too_few_or_duplicate_queries_rejected(self):
        for data in (rows(29), rows()+[rows()[0]]):
            with self.assertRaises(ValueError):
                calibrate(data)

    def test_nonfinite_and_out_of_range_data_rejected(self):
        data = rows()
        data[0]['primary']['ms'] = math.nan
        with self.assertRaises(ValueError):
            calibrate(data)
        data = rows()
        data[0]['features'] = [2,0,0]
        with self.assertRaises(ValueError):
            calibrate(data)

    def test_bootstrap_constant_and_invalid_inputs(self):
        self.assertAlmostEqual(bootstrap_lower([-.02]*10, repetitions=100), -.02)
        for data in ([], [math.nan], [2]):
            with self.assertRaises(ValueError):
                bootstrap_lower(data)

    def test_always_fusion_and_empty_results_never_accept(self):
        row = rows()[0]
        self.assertFalse(accepts(row, dict(threshold=None, weights=[1,0,0])))
        row['primary']['ids'] = []
        self.assertFalse(accepts(row, dict(threshold=0, weights=[1,0,0])))

    def test_self_exclusion_refills_and_reranks(self):
        class Backend:
            def search(self, query, *, top_k):
                self.depth = top_k
                return [SearchResult(Chunk(str(i), str(i), 'text'), 10-i, i+1, 'test')
                        for i in range(top_k)]
        backend = Backend()
        result = ExcludeSelf(backend).search(Query('text', {'evaluation_query_id': '1'}), top_k=3)
        self.assertEqual(backend.depth, 4)
        self.assertEqual([r.chunk.id for r in result], ['0','2','3'])
        self.assertEqual([r.rank for r in result], [1,2,3])


if __name__ == '__main__':
    unittest.main()
