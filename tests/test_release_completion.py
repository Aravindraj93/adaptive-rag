import concurrent.futures
import json
import math
import tempfile
import unittest
from pathlib import Path
from adaptive_rag import CachedRetriever, Chunk, Query, SearchResult, TelemetryCollector
from adaptive_rag.persistence import IndexFormatError, read_manifest, write_manifest


class Backend:
    def __init__(self):
        self.calls = 0
    def search(self, query, *, top_k=5):
        self.calls += 1
        return [SearchResult(Chunk('a','doc','generic text', {'nested': [1]}), 1., 1, 'baseline')]


class CompletionTests(unittest.TestCase):
    def cache(self, **kw):
        self.backend = Backend()
        self.version = 1
        return CachedRetriever(self.backend, revision=lambda: self.version, scope='tenant-a', **kw)

    def test_exact_hit_and_revision_invalidation(self):
        cache = self.cache()
        a = cache.search('query')
        self.assertEqual(cache.search('query'), a)
        self.assertEqual(self.backend.calls, 1)
        self.version += 1
        cache.search('query')
        self.assertEqual(self.backend.calls, 2)

    def test_query_type_topk_and_metadata_isolation(self):
        cache = self.cache()
        for query in ['q', Query('q'), Query('q', {'tenant':'b'}),
                      Query('q', {'key':None}), Query('q', {'key':1}), Query('q', {'key':True})]:
            cache.search(query)
        cache.search('q', top_k=2)
        self.assertEqual(self.backend.calls, 7)

    def test_nested_results_cannot_mutate_cache(self):
        cache = self.cache()
        cache.search('q')[0].chunk.metadata['nested'].append(2)
        got = cache.search('q')
        self.assertEqual(got[0].chunk.metadata['nested'], [1])
        got[0].chunk.metadata['nested'].append(3)
        self.assertEqual(cache.search('q')[0].chunk.metadata['nested'], [1])

    def test_lru_and_byte_bound(self):
        cache = self.cache(max_entries=2)
        for q in ['a','b','a','c']:
            cache.search(q)
        self.assertEqual(cache.info().entries, 2)
        cache.search('b')
        self.assertEqual(self.backend.calls, 4)
        tiny = self.cache(max_bytes=1)
        tiny.search('q')
        self.assertEqual(tiny.info().entries, 0)

    def test_unsupported_keys_bypass_without_collisions(self):
        cache = self.cache()
        for q in [Query('q', {'x':(1,2)}), Query('q', {'x':[1,2]}),
                  Query('q', {1:'v'}), Query('q', {'1':'v'})]:
            cache.search(q)
        self.assertEqual(self.backend.calls, 4)
        self.assertEqual(cache.info().bypasses, 2)

    def test_revision_change_during_miss_is_not_cached(self):
        cache = self.cache()
        original = self.backend.search
        def change(*a, **kw):
            self.version += 1
            return original(*a, **kw)
        self.backend.search = change
        cache.search('q')
        self.assertEqual(cache.info().entries, 0)

    def test_concurrent_duplicate_miss_is_single_flight(self):
        cache = self.cache()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            result = list(pool.map(lambda _: cache.search('q'), range(100)))
        self.assertTrue(all(r == result[0] for r in result))
        self.assertEqual(self.backend.calls, 1)
        self.assertEqual(cache.info().hits, 99)

    def test_clear_invalidates(self):
        cache = self.cache()
        cache.search('q')
        cache.clear()
        cache.search('q')
        self.assertEqual(self.backend.calls, 2)

    def test_cache_validation(self):
        for kw in ({'max_entries':0}, {'max_bytes':0}, {'max_entries':True}):
            with self.assertRaises(ValueError):
                self.cache(**kw)
        cache = self.cache()
        self.version = None
        with self.assertRaises(TypeError):
            cache.search('q')

    def test_telemetry_bounded_and_unbounded_opt_in(self):
        metrics = TelemetryCollector(max_events=3)
        for _ in range(10):
            metrics.record('search', 1)
        self.assertEqual(len(metrics.snapshot()), 3)
        self.assertEqual(metrics.dropped_events, 7)
        unlimited = TelemetryCollector(max_events=None)
        for _ in range(1500):
            unlimited.record('search', 1)
        self.assertEqual(len(unlimited.snapshot()), 1500)

    def test_telemetry_invalid_duration(self):
        for value in (-1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                TelemetryCollector().record('bad', value)

    def test_strict_manifests(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'manifest.json'
            for text in ('{"payload":{},"payload":{}}', '{"payload":NaN}', '{"payload":Infinity}'):
                path.write_text(text)
                with self.assertRaises(IndexFormatError):
                    read_manifest(path)
            write_manifest(path, {'a':1})
            with self.assertRaises(IndexFormatError):
                read_manifest(path, max_bytes=1)
            envelope = json.loads(path.read_text())
            envelope['schema_version'] = True
            path.write_text(json.dumps(envelope))
            with self.assertRaises(IndexFormatError):
                read_manifest(path)
            with self.assertRaises(TypeError):
                write_manifest(path, {'x': math.nan})


if __name__ == '__main__':
    unittest.main()
