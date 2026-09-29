import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from adaptive_rag.models import Chunk, Query, SearchResult
from adaptive_rag.retrievers.bm25 import BM25Retriever
from adaptive_rag.semantic_cache import SemanticCachedRetriever


class Backend:
    def __init__(self):
        self.calls = 0
        self.version = 0

    def search(self, query, *, top_k=5, **kwargs):
        self.calls += 1
        text = query.text if isinstance(query, Query) else query
        return [SearchResult(Chunk(f'{text}-{i}', 'd', text,
                    {'nested': [1], 'tenant': kwargs.get('tenant'), 'v': self.version}),
                    1.0, i+1, 'test') for i in range(top_k)]


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()

    def cache(self, **kwargs):
        return SemanticCachedRetriever(self.backend, revision=lambda: self.backend.version,
                                       scope='test', **kwargs)

    def test_default_exact_no_embedder(self):
        cache = self.cache()
        cache.search('a'); cache.search('a'); cache.search('b')
        self.assertEqual(self.backend.calls, 2)
        self.assertEqual(cache.info().exact_hits, 1)

    def test_top_k_and_tenants(self):
        cache = self.cache()
        cache.search('a', top_k=1, tenant='A')
        self.assertEqual(len(cache.search('a', top_k=3, tenant='A')), 3)
        self.assertEqual(cache.search('a', top_k=1, tenant='B')[0].chunk.metadata['tenant'], 'B')
        self.assertEqual(self.backend.calls, 3)

    def test_query_metadata_isolation(self):
        cache = self.cache()
        for q in ['a', Query('a'), Query('a', {'tenant':'A'}), Query('a', {'tenant':'B'})]:
            cache.search(q)
        self.assertEqual(self.backend.calls, 4)

    def test_revision_and_copy_isolation(self):
        cache = self.cache()
        cache.search('a')[0].chunk.metadata['nested'].append(2)
        self.assertEqual(cache.search('a')[0].chunk.metadata['nested'], [1])
        self.backend.version += 1
        self.assertEqual(cache.search('a')[0].chunk.metadata['v'], 1)
        self.assertEqual(cache.info().entries, 1)

    def test_semantic_requires_validator(self):
        with self.assertRaises(ValueError):
            self.cache(mode='semantic', embedder=lambda _: [1, 0])

    def test_semantic_opt_in_and_threshold_one(self):
        cache = self.cache(mode='semantic', embedder=lambda _: [1, 0], threshold=1,
                           reuse_validator=lambda a,b: {a,b} == {'a', 'b'})
        cache.search('a')
        self.assertEqual(cache.search('b')[0].chunk.id, 'a-0')
        cache.search('not a')
        self.assertEqual(self.backend.calls, 2)
        self.assertEqual(cache.info().semantic_hits, 1)

    def test_shadow_serves_fresh_results(self):
        cache = self.cache(mode='shadow', embedder=lambda _: [1, 0])
        cache.search('a')
        self.assertEqual(cache.search('b')[0].chunk.id, 'b-0')
        self.assertEqual(cache.info().shadow_matches, 1)
        self.assertEqual(cache.info().shadow_disagreements, 1)

    def test_semantic_contexts_isolated(self):
        cache = self.cache(mode='semantic', embedder=lambda _: [1,0], reuse_validator=lambda a,b: True)
        cache.search(Query('a', {'filter': 1}), tenant='A')
        cache.search(Query('b', {'filter': 2}), tenant='A')
        cache.search(Query('c', {'filter': 2}), tenant='B')
        cache.search(Query('d', {'filter': 2}), tenant='B', top_k=1)
        self.assertEqual(self.backend.calls, 4)

    def test_ttl_and_limits(self):
        with patch('adaptive_rag.semantic_cache.monotonic', return_value=0):
            cache = self.cache(ttl_seconds=1, max_entries=1)
            cache.search('a'); cache.search('b')
            self.assertEqual(cache.info().entries, 1)
        with patch('adaptive_rag.semantic_cache.monotonic', return_value=1):
            self.assertEqual(cache.info().entries, 0)
            cache.search('b')
        self.assertEqual(self.backend.calls, 3)
        tiny = self.cache(max_bytes=1)
        tiny.search('a')
        self.assertEqual(tiny.info().entries, 0)
        self.assertEqual(tiny.info().bypasses, 1)

    def test_bad_configuration_and_embeddings(self):
        for kw in [{'threshold':float('nan')}, {'max_entries':0}, {'ttl_seconds':0}, {'mode':'x'}]:
            with self.assertRaises(ValueError): self.cache(**kw)
        for vector in [[], [0,0], [float('nan')], [float('inf')]]:
            cache = self.cache(mode='shadow', embedder=lambda _, v=vector: v)
            with self.assertRaises(ValueError): cache.search('a')

    def test_embedding_once_per_exact_miss(self):
        calls = []
        cache = self.cache(mode='shadow', embedder=lambda text: calls.append(text) or [1,0])
        cache.search('a'); cache.search('a')
        self.assertEqual(calls, ['a'])

    def test_unsupported_metadata_bypasses(self):
        cache = self.cache()
        cache.search(Query('a', {'not_json': object()}))
        self.assertEqual(cache.info().bypasses, 1)
        self.assertEqual(cache.info().entries, 0)

    def test_revision_change_during_search_not_cached(self):
        original = self.backend.search
        def changing(*args, **kwargs):
            rows = original(*args, **kwargs)
            self.backend.version += 1
            return rows
        self.backend.search = changing
        cache = self.cache()
        cache.search('a')
        self.assertEqual(cache.info().entries, 0)

    def test_single_flight_concurrent_calls(self):
        cache = self.cache()
        with ThreadPoolExecutor(max_workers=8) as executor:
            rows = list(executor.map(cache.search, ['a'] * 100))
        self.assertEqual(self.backend.calls, 1)
        self.assertEqual(len(rows), 100)

    def test_real_backend(self):
        backend = BM25Retriever()
        backend.add([Chunk('c','d','hello world')])
        cache = SemanticCachedRetriever(backend, revision=lambda: 0, scope='real')
        self.assertEqual(cache.search('hello'), cache.search('hello'))
        self.assertEqual(cache.info().exact_hits, 1)

    def test_exact_mode_keeps_hard_negatives_separate(self):
        cache = self.cache()
        queries = ['enable feature', 'disable feature', 'limit 3', 'limit 30',
                   'before 2025', 'before 2026', 'REQ_001', 'REQ_002']
        for text in queries:
            self.assertEqual(cache.search(text)[0].chunk.text, text)
        self.assertEqual(self.backend.calls, len(queries))

    def test_shadow_exposes_hard_negative_disagreement(self):
        # Deliberately indistinguishable embeddings: this is a safety/control
        # test, not evidence about a real model's semantic quality.
        cache = self.cache(mode='shadow', embedder=lambda _: [1, 0])
        cache.search('enable feature')
        self.assertEqual(cache.search('disable feature')[0].chunk.text, 'disable feature')
        self.assertEqual(cache.info().shadow_disagreements, 1)

    def test_expiry_during_semantic_embedding_does_not_serve_old_rows(self):
        clock = [0]
        def embed(text):
            if text == 'b': clock[0] = 2
            return [1, 0]
        with patch('adaptive_rag.semantic_cache.monotonic', side_effect=lambda: clock[0]):
            cache = self.cache(mode='semantic', embedder=embed, ttl_seconds=1,
                               reuse_validator=lambda a,b: True)
            cache.search('a')
            self.assertEqual(cache.search('b')[0].chunk.text, 'b')
            self.assertEqual(cache.info().semantic_hits, 0)

    def test_dimension_change_does_not_match(self):
        cache = self.cache(mode='shadow', embedder=lambda text: [1,0] if text == 'a' else [1,0,0])
        cache.search('a'); cache.search('b')
        self.assertEqual(cache.info().shadow_matches, 0)

    def test_concurrent_tenants_remain_isolated(self):
        cache = self.cache()
        def search(i):
            tenant = str(i % 8)
            row = cache.search('same', tenant=tenant)[0]
            return row.chunk.metadata['tenant'] == tenant
        with ThreadPoolExecutor(max_workers=8) as executor:
            self.assertTrue(all(executor.map(search, range(400))))
        self.assertEqual(self.backend.calls, 8)

    def test_public_exports(self):
        import adaptive_rag
        self.assertTrue(all(hasattr(adaptive_rag, name) for name in adaptive_rag.__all__))
