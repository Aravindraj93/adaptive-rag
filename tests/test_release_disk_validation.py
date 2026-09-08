import tempfile
import unittest
from pathlib import Path
from adaptive_rag import Chunk, HashingEmbedder, MMapDenseRetriever, MMapBM25Retriever
from adaptive_rag.persistence import read_manifest, write_manifest, IndexFormatError


class DiskValidationTests(unittest.TestCase):
    def test_sparse_out_of_file_range_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'index'
            MMapBM25Retriever.build(path, [Chunk('a','a','shared text')]).close()
            file = path/'manifest.json'
            value = read_manifest(file)
            value['chunks'][0]['offset'] = -1
            write_manifest(file, value)
            with self.assertRaises(IndexFormatError):
                MMapBM25Retriever(path)

    def test_dense_out_of_file_range_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'index'
            embedder = HashingEmbedder()
            MMapDenseRetriever.build(path, [Chunk('a','a','shared text')], embedder=embedder).close()
            file = path/'manifest.json'
            value = read_manifest(file)
            value['chunks'][0]['size'] = 10**12
            write_manifest(file, value)
            with self.assertRaises(IndexFormatError):
                MMapDenseRetriever(path, embedder=embedder)

    def test_dense_missing_second_file_closes_first(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'index'
            embedder = HashingEmbedder()
            MMapDenseRetriever.build(path, [Chunk('a','a','shared text')], embedder=embedder).close()
            (path/'vectors.f32').unlink()
            with self.assertRaises(OSError):
                MMapDenseRetriever(path, embedder=embedder, verify_checksums=False)
            (path/'chunks.jsonl').rename(path/'renamed.jsonl')

    def test_sparse_nonpositive_parameters_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'index'
            MMapBM25Retriever.build(path, [Chunk('a','a','shared text')]).close()
            file = path/'manifest.json'
            value = read_manifest(file)
            value['configuration']['k1'] = -1
            write_manifest(file,value)
            with self.assertRaises(IndexFormatError):
                MMapBM25Retriever(path)

    def test_overflowed_json_number_has_format_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'manifest.json'
            path.write_text('{"schema_version":1,"checksum_algorithm":"sha256","checksum":"x","payload":{"x":1e999}}')
            with self.assertRaises(IndexFormatError):
                read_manifest(path)


if __name__ == '__main__':
    unittest.main()
