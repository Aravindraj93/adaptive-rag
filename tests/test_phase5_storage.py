import tempfile
import unittest
from pathlib import Path

from adaptive_rag import BM25Retriever, Chunk, HashingEmbedder, MetadataFilter
from adaptive_rag.retrievers.dense import DenseRetriever
from adaptive_rag.retrievers.mmap_bm25 import MMapBM25Retriever
from adaptive_rag.retrievers.mmap_dense import MMapDenseRetriever
from adaptive_rag.retrievers.segmented_bm25 import SegmentedBM25Index, SegmentedBM25Retriever


class NativeFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            Chunk("north", "d", "shared apple policy", {"region": "north"}),
            Chunk("south", "d", "shared apple policy", {"region": "south"}),
        ]
        self.where = MetadataFilter(equals={"region": "south"})

    def test_in_memory_bm25_filters_before_ranking(self) -> None:
        retriever = BM25Retriever()
        retriever.add(self.chunks)
        self.assertEqual(retriever.search("apple policy", where=self.where)[0].chunk.id, "south")

    def test_dense_filters_before_scoring(self) -> None:
        retriever = DenseRetriever(HashingEmbedder(dimensions=32))
        retriever.add(self.chunks)
        self.assertEqual(retriever.search("apple policy", where=self.where)[0].chunk.id, "south")

    def test_mmap_bm25_filters_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with MMapBM25Retriever.build(Path(directory) / "index", self.chunks) as retriever:
                result = retriever.search("apple policy", where=self.where)[0]
                self.assertEqual(result.chunk.id, "south")


class ExternalStagingTests(unittest.TestCase):
    def test_builder_leaves_only_runtime_index_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index"
            chunks = (Chunk(str(i), "d", f"common marker{i}") for i in range(50))
            MMapBM25Retriever.build(path, chunks).close()
            self.assertEqual(
                {item.name for item in path.iterdir()},
                {"manifest.json", "chunks.jsonl", "postings.jsonl", "facets.sqlite"},
            )


class MMapDenseTests(unittest.TestCase):
    def test_binary_index_matches_in_memory_dense_ranking_and_filters(self) -> None:
        embedder = HashingEmbedder(dimensions=32)
        chunks = [
            Chunk("apple", "d", "fresh apple orchard", {"kind": "fruit"}),
            Chunk("ocean", "d", "deep ocean current", {"kind": "water"}),
        ]
        baseline = DenseRetriever(embedder)
        baseline.add(chunks)
        expected = [result.chunk.id for result in baseline.search("apple orchard")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dense"
            with MMapDenseRetriever.build(path, chunks, embedder=embedder, batch_size=1) as retriever:
                actual = [result.chunk.id for result in retriever.search("apple orchard")]
                self.assertEqual(actual, expected)
                filtered = retriever.search(
                    "apple orchard", where=MetadataFilter(equals={"kind": "fruit"})
                )
                self.assertEqual(filtered[0].chunk.id, "apple")
            self.assertTrue(retriever.closed)

    def test_vector_tampering_is_rejected(self) -> None:
        embedder = HashingEmbedder(dimensions=16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dense"
            retriever = MMapDenseRetriever.build(
                path, [Chunk("a", "d", "apple")], embedder=embedder
            )
            retriever.close()
            with (path / "vectors.f32").open("ab") as stream:
                stream.write(b"bad")
            with self.assertRaisesRegex(ValueError, "checksum"):
                MMapDenseRetriever(path, embedder=embedder)


class SegmentedIndexTests(unittest.TestCase):
    def test_append_delete_reopen_and_compact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segments"
            manager = SegmentedBM25Index.create(
                path,
                [Chunk("apple", "d", "fresh apple orchard")],
            )
            self.assertEqual(manager.append([Chunk("ocean", "d", "deep ocean current")]), 1)
            with SegmentedBM25Retriever(path) as retriever:
                self.assertEqual(retriever.search("ocean")[0].chunk.id, "ocean")
            self.assertEqual(manager.delete(["apple"]), 1)
            with SegmentedBM25Retriever(path) as retriever:
                self.assertEqual(retriever.search("apple"), [])
                self.assertEqual(len(list(retriever.iter_chunks())), 1)
            self.assertEqual(manager.compact(), 1)
            with SegmentedBM25Retriever(path) as retriever:
                self.assertEqual(len(retriever._segments), 1)
                self.assertEqual(retriever.search("ocean")[0].chunk.id, "ocean")

    def test_deleted_id_cannot_be_reused_before_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segments"
            manager = SegmentedBM25Index.create(
                path,
                [Chunk("a", "d", "alpha"), Chunk("b", "d", "beta")],
            )
            manager.delete(["a"])
            with self.assertRaisesRegex(ValueError, "already exist"):
                manager.append([Chunk("a", "d", "replacement")])

    def test_segment_append_refuses_existing_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segments"
            manager = SegmentedBM25Index.create(path, [Chunk("a", "d", "text")])
            with self.assertRaisesRegex(ValueError, "already exist"):
                manager.append([Chunk("a", "d", "replacement")])


if __name__ == "__main__":
    unittest.main()
