import unittest
from adaptive_rag.comparison import ComparisonCase, compare_retrievers
from adaptive_rag.models import Chunk, SearchResult


class Fixed:
    def __init__(self, ids): self.ids = ids
    def search(self, query, *, top_k=5):
        return [SearchResult(Chunk(i,'d','text'), 1., rank, 'fixture')
                for rank, i in enumerate(self.ids[:top_k], 1)]


class ComparisonTests(unittest.TestCase):
    def test_quality_regression(self):
        report = compare_retrievers(Fixed(['a']), Fixed(['b']),
            [ComparisonCase('query', frozenset({'a'}))], repetitions=2)
        self.assertEqual(report['quality_regressions'], 2)
        self.assertEqual(report['summary']['baseline']['ndcg_at_k'], 1)
        self.assertEqual(report['summary']['candidate']['recall_at_k'], 0)

    def test_unlabelled_is_not_quality(self):
        report = compare_retrievers(Fixed(['a']), Fixed(['a']), [ComparisonCase('private query')])
        self.assertFalse(report['quality_evaluated'])
        self.assertIsNone(report['summary']['baseline']['recall_at_k'])
        self.assertIsNone(report['per_query'][0]['quality_regression'])
        self.assertIsNone(report['summary']['baseline']['repeated_passes'])
        self.assertNotIn('private query', str(report))

    def test_binary_ndcg(self):
        report = compare_retrievers(Fixed(['b','a']), Fixed(['a','b']),
                                   [ComparisonCase('q', frozenset({'a'}))])
        self.assertAlmostEqual(report['summary']['baseline']['ndcg_at_k'], 0.6309297536)

    def test_validation(self):
        with self.assertRaises(ValueError):
            compare_retrievers(Fixed([]), Fixed([]), [])
        with self.assertRaises(ValueError):
            compare_retrievers(Fixed(['a','a']), Fixed([]), [ComparisonCase('q')])
