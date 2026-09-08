import tempfile
import unittest
from pathlib import Path
from adaptive_rag import Chunk, MetadataFilter, MMapBM25Retriever
from adaptive_rag.persistence import read_manifest, write_manifest


class LegacySparseTests(unittest.TestCase):
    def test_manifest_without_facets_uses_scan_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'index'
            MMapBM25Retriever.build(path, [
                Chunk('a', 'd', 'shared text', {'region': 'a'}),
                Chunk('b', 'd', 'shared text', {'region': 'b'}),
            ]).close()
            manifest = path / 'manifest.json'
            payload = read_manifest(manifest)
            del payload['file_checksums']['facets.sqlite']
            write_manifest(manifest, payload)
            with MMapBM25Retriever(path) as reader:
                result = reader.search('shared', where=MetadataFilter(equals={'region': 'b'}))
                self.assertEqual(result[0].chunk.id, 'b')
