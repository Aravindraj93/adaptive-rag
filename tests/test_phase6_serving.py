import tempfile
import unittest
from pathlib import Path

from adaptive_rag import Chunk, HashingEmbedder, MetadataFilter
from adaptive_rag.async_index import AsyncSegmentCoordinator
from adaptive_rag.retrievers.lsh_dense import LSHDenseRetriever
from adaptive_rag.retrievers.mmap_bm25 import MMapBM25Retriever
from adaptive_rag.retrievers.segmented_bm25 import SegmentedBM25Index, SegmentedBM25Retriever


class FacetPushdownTests(unittest.TestCase):
    def test_filter_reduces_candidates_before_scoring(self) -> None:
        chunks = [
            Chunk("north", "d", "shared policy", {"region": "north", "year": 2025}),
            Chunk("south", "d", "shared policy", {"region": "south", "year": 2026}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index"
            with MMapBM25Retriever.build(path, chunks) as retriever:
                results = retriever.search(
                    "shared policy",
                    where=MetadataFilter(
                        any_of={"region": frozenset({"south"})},
                        exists=frozenset({"year"}),
                    ),
                )
                self.assertEqual(results[0].chunk.id, "south")
                self.assertEqual(retriever.last_stats.candidates, 1)
            self.assertTrue((path / "facets.sqlite").is_file())


class AsyncCoordinatorTests(unittest.TestCase):
    def test_jobs_are_serialized_in_submission_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segments"
            SegmentedBM25Index.create(path, [Chunk("first", "d", "first entry")])
            with AsyncSegmentCoordinator(path) as coordinator:
                appended = coordinator.append([Chunk("second", "d", "second entry")])
                deleted = coordinator.delete(["first"])
                compacted = coordinator.compact()
                self.assertEqual(appended.result(timeout=10), 1)
                self.assertEqual(deleted.result(timeout=10), 1)
                self.assertEqual(compacted.result(timeout=10), 1)
            with SegmentedBM25Retriever(path) as retriever:
                self.assertEqual([chunk.id for chunk in retriever.iter_chunks()], ["second"])

    def test_closed_coordinator_rejects_new_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segments"
            SegmentedBM25Index.create(path, [Chunk("first", "d", "first entry")])
            coordinator = AsyncSegmentCoordinator(path)
            coordinator.close()
            with self.assertRaises(RuntimeError):
                coordinator.compact()


class LSHDenseTests(unittest.TestCase):
    def test_near_identical_query_retrieves_expected_chunk(self) -> None:
        embedder = HashingEmbedder(dimensions=64)
        retriever = LSHDenseRetriever(
            embedder, dimensions=64, tables=8, bits=8, probe_radius=1, seed=7
        )
        retriever.add([
            Chunk("apple", "d", "fresh apple orchard fruit"),
            Chunk("ocean", "d", "deep ocean current water"),
            Chunk("policy", "d", "account refund policy request"),
        ])
        self.assertEqual(retriever.search("fresh apple orchard")[0].chunk.id, "apple")
        self.assertLessEqual(retriever.last_stats.candidates, 3)

    def test_lsh_respects_native_metadata_filter(self) -> None:
        embedder = HashingEmbedder(dimensions=32)
        retriever = LSHDenseRetriever(
            embedder, dimensions=32, tables=12, bits=4, probe_radius=1, seed=2
        )
        retriever.add([
            Chunk("a", "d", "shared apple", {"group": "a"}),
            Chunk("b", "d", "shared apple", {"group": "b"}),
        ])
        result = retriever.search(
            "shared apple", where=MetadataFilter(equals={"group": "b"})
        )
        self.assertEqual(result[0].chunk.id, "b")


if __name__ == "__main__":
    unittest.main()
