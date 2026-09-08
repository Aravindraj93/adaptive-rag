import tempfile
import unittest
from pathlib import Path

from adaptive_rag import BM25Retriever, Chunk, IndexFormatError
from adaptive_rag.retrievers.mmap_bm25 import MMapBM25Retriever


class MMapBM25Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            Chunk("a", "one", "Fresh apples grow in an orchard", {"group": "fruit"}),
            Chunk("b", "two", "Ocean currents move deep water"),
            Chunk("c", "three", "An apple pie uses fresh fruit"),
        ]

    def test_disk_ranking_matches_in_memory_baseline(self) -> None:
        baseline = BM25Retriever()
        baseline.add(self.chunks)
        expected = [result.chunk.id for result in baseline.search("fresh apples", top_k=3)]
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "index"
            with MMapBM25Retriever.build(index_path, self.chunks) as retriever:
                actual = [result.chunk.id for result in retriever.search("fresh apples", top_k=3)]
                self.assertEqual(actual, expected)
                self.assertEqual(retriever.search("orchard")[0].chunk.metadata["group"], "fruit")
            self.assertTrue(retriever.closed)
            with self.assertRaises(RuntimeError):
                retriever.search("apple")

    def test_data_file_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "index"
            retriever = MMapBM25Retriever.build(index_path, self.chunks)
            retriever.close()
            with (index_path / MMapBM25Retriever.CHUNKS).open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(IndexFormatError, "checksum"):
                MMapBM25Retriever(index_path)

    def test_existing_directory_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "index"
            first = MMapBM25Retriever.build(index_path, self.chunks)
            first.close()
            with self.assertRaises(FileExistsError):
                MMapBM25Retriever.build(index_path, self.chunks)
            second = MMapBM25Retriever.build(index_path, self.chunks[:1], overwrite=True)
            self.assertEqual(len(second), 1)
            second.close()


if __name__ == "__main__":
    unittest.main()

