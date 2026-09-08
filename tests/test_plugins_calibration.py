import unittest

from adaptive_rag import BM25Retriever, Chunk, Document, Query
from adaptive_rag.benchmark import BenchmarkCase
from adaptive_rag.calibration import calibrate_confidence
from adaptive_rag.plugins import PluginPipeline


class LengthEnricher:
    name = "length"

    def enrich_document(self, document):
        return {"character_count": len(document.text)}


class IntentAnalyzer:
    name = "intent"

    def analyze_query(self, query):
        return {"question": query.text.endswith("?")}


class ConflictingEnricher:
    name = "conflict"

    def enrich_document(self, document):
        return {"source": "replacement"}


class PluginTests(unittest.TestCase):
    def test_pipeline_adds_generic_metadata(self) -> None:
        pipeline = PluginPipeline(
            document_enrichers=[LengthEnricher()], query_analyzers=[IntentAnalyzer()]
        )
        document = pipeline.enrich_documents([Document("d", "hello")])[0]
        query = pipeline.analyze_query(Query("Where?"))
        self.assertEqual(document.metadata["character_count"], 5)
        self.assertTrue(query.metadata["question"])

    def test_plugins_cannot_silently_overwrite_metadata(self) -> None:
        pipeline = PluginPipeline(document_enrichers=[ConflictingEnricher()])
        with self.assertRaisesRegex(ValueError, "overwrite"):
            pipeline.enrich_documents([Document("d", "text", {"source": "original"})])


class CalibrationTests(unittest.TestCase):
    def test_calibration_separates_strong_and_weak_retrievals(self) -> None:
        retriever = BM25Retriever()
        retriever.add([Chunk("apple", "d", "fresh apple orchard")])
        cases = [
            BenchmarkCase(Query("fresh apple orchard"), frozenset({"apple"})),
            BenchmarkCase(Query("unknown terms plus apple"), frozenset()),
        ]
        report = calibrate_confidence(retriever, cases)
        self.assertEqual(report.f1, 1.0)
        self.assertEqual(report.precision, 1.0)
        self.assertEqual(report.recall, 1.0)
        self.assertEqual(report.acceptance_rate, 0.5)


if __name__ == "__main__":
    unittest.main()

