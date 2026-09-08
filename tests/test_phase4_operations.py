import tempfile
import unittest
from pathlib import Path

from adaptive_rag import BM25Retriever, Chunk
from adaptive_rag.embedders import HashingEmbedder
from adaptive_rag.filtering import MetadataFilter, MetadataFilteredRetriever
from adaptive_rag.index_lifecycle import MMapIndexManager
from adaptive_rag.retrievers.dense import DenseRetriever
from adaptive_rag.retrievers.mmap_bm25 import MMapBM25Retriever
from adaptive_rag.routing import EscalatingRetriever
from adaptive_rag.scale_benchmark import run_scale_benchmark


class HashingAndDensePersistenceTests(unittest.TestCase):
    def test_hashing_embedder_is_deterministic(self) -> None:
        embedder = HashingEmbedder(dimensions=32)
        self.assertEqual(embedder(["Fresh apples"]), embedder(["Fresh apples"]))
        self.assertNotEqual(embedder(["Fresh apples"]), embedder(["Deep ocean"]))

    def test_dense_round_trip_does_not_reembed_stored_chunks(self) -> None:
        embedder = HashingEmbedder(dimensions=32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dense.json"
            original = DenseRetriever(embedder)
            original.add([
                Chunk("apple", "d", "fresh apple orchard", {"kind": "fruit"}),
                Chunk("ocean", "d", "deep ocean current"),
            ])
            original.save(path)
            restored = DenseRetriever.load(path, embedder=embedder)
            result = restored.search("apple orchard")[0]
            self.assertEqual(result.chunk.id, "apple")
            self.assertEqual(result.chunk.metadata["kind"], "fruit")


class FilterAndRoutingTests(unittest.TestCase):
    def test_metadata_filter_supports_equals_membership_and_exists(self) -> None:
        baseline = BM25Retriever()
        baseline.add([
            Chunk("a", "d", "shared policy", {"region": "north", "year": 2025}),
            Chunk("b", "d", "shared policy", {"region": "south", "year": 2026}),
        ])
        filtered = MetadataFilteredRetriever(
            baseline,
            MetadataFilter(
                equals={"region": "south"},
                any_of={"year": frozenset({2026, 2027})},
                exists=frozenset({"region"}),
            ),
        )
        self.assertEqual(filtered.search("shared policy")[0].chunk.id, "b")

    def test_router_uses_primary_then_fallback_on_no_match(self) -> None:
        primary = BM25Retriever()
        primary.add([Chunk("apple", "d", "fresh apple orchard")])
        fallback = BM25Retriever()
        fallback.add([Chunk("volcano", "d", "volcano magma eruption")])
        router = EscalatingRetriever(primary, fallback)
        self.assertEqual(router.search("fresh apple orchard")[0].chunk.id, "apple")
        self.assertEqual(router.last_route.route, "primary")
        self.assertEqual(router.search("volcano eruption")[0].chunk.id, "volcano")
        self.assertEqual(router.last_route.route, "fallback")


class LifecycleTests(unittest.TestCase):
    def test_append_upsert_delete_and_compact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index"
            source = (Chunk(str(number), "d", f"entry marker{number}") for number in range(3))
            MMapBM25Retriever.build(path, source).close()
            manager = MMapIndexManager(path)

            appended = manager.append([Chunk("3", "d", "entry marker3")])
            self.assertEqual((appended.before, appended.added, appended.after), (3, 1, 4))
            updated = manager.update(upserts=[Chunk("3", "d", "replacement special")])
            self.assertEqual(updated.replaced, 1)
            deleted = manager.delete(["0"])
            self.assertEqual(deleted.deleted, 1)
            compacted = manager.compact()
            self.assertEqual(compacted.before, compacted.after)

            with MMapBM25Retriever(path) as retriever:
                self.assertEqual(len(retriever), 3)
                self.assertEqual(retriever.search("special")[0].chunk.id, "3")
                self.assertEqual(retriever.search("marker0"), [])

    def test_append_refuses_existing_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index"
            MMapBM25Retriever.build(path, [Chunk("a", "d", "text")]).close()
            with self.assertRaisesRegex(ValueError, "already exist"):
                MMapIndexManager(path).append([Chunk("a", "d", "replacement")])


class ScaleBenchmarkTests(unittest.TestCase):
    def test_small_scale_run_reports_quality_memory_and_startup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_scale_benchmark(
                documents=100,
                queries=10,
                index_path=Path(directory) / "index",
            )
        self.assertEqual(report["in_memory"]["top1_accuracy"], 1.0)
        self.assertEqual(report["memory_mapped"]["top1_accuracy"], 1.0)
        self.assertGreater(report["in_memory"]["retained_python_bytes"], 0)
        self.assertGreater(report["in_memory"]["peak_python_bytes"], 0)
        self.assertGreater(report["memory_mapped"]["index_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
