import tempfile
import unittest
from pathlib import Path

from adaptive_rag import Chunk, MetadataFilter, MMapBM25Retriever
from adaptive_rag import SegmentedBM25Index, SegmentedBM25Retriever
from adaptive_rag.persistence import read_manifest, write_manifest, IndexFormatError
from adaptive_rag.retrievers.dense import DenseRetriever


class ReleaseTests(unittest.TestCase):
    def test_cleanup_preserves_open_reader_then_reclaims(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'index'
            manager = SegmentedBM25Index.create(path, [Chunk('a', 'd', 'shared text')])
            with SegmentedBM25Retriever(path) as old:
                manager.compact()
                self.assertEqual(manager.cleanup(dry_run=False), [])
                self.assertEqual(old.search('shared')[0].chunk.id, 'a')
            candidates = manager.cleanup()
            self.assertEqual(len(candidates), 1)
            self.assertTrue((path / candidates[0]).exists())
            self.assertEqual(manager.cleanup(dry_run=False), candidates)
            self.assertFalse((path / candidates[0]).exists())
            with SegmentedBM25Retriever(path) as reader:
                self.assertEqual(reader.search('shared')[0].chunk.id, 'a')

    def test_segment_manifest_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'index'
            SegmentedBM25Index.create(path, [Chunk('a', 'd', 'text')])
            manifest = path / 'segments.json'
            payload = read_manifest(manifest)
            payload['segments'] = ['../outside']
            write_manifest(manifest, payload)
            with self.assertRaises(IndexFormatError):
                SegmentedBM25Retriever(path)

    def test_filter_null_and_types_agree_with_facets(self):
        chunks = [Chunk(str(i), 'd', 'shared text', meta) for i, meta in enumerate([
            {}, {'key': None}, {'key': 1}, {'key': True}, {'key': 1.0}, {'key': [1, 2]}
        ])]
        with tempfile.TemporaryDirectory() as directory:
            with MMapBM25Retriever.build(Path(directory) / 'index', chunks) as reader:
                for value in (None, 1, True, 1.0, [1, 2]):
                    where = MetadataFilter(equals={'key': value})
                    expected = {c.id for c in chunks if where.matches(c.metadata)}
                    actual = {r.chunk.id for r in reader.search('shared', top_k=10, where=where)}
                    self.assertEqual(actual, expected)
                self.assertFalse(MetadataFilter(equals={'key': None}).matches({}))
                self.assertFalse(MetadataFilter(any_of={'key': frozenset({1})}).matches({'key': [1]}))

    def test_nonfinite_embeddings_rejected(self):
        for value in (float('nan'), float('inf')):
            index = DenseRetriever(lambda texts: [[value, 1] for _ in texts])
            with self.assertRaises(ValueError):
                index.add([Chunk('a', 'd', 'text')])
