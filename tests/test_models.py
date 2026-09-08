import unittest

from adaptive_rag import Chunk, Document, Query, SearchResult


class ModelTests(unittest.TestCase):
    def test_metadata_is_copied_and_read_only(self) -> None:
        source = {"language": "en"}
        document = Document("doc", "text", source)
        source["language"] = "fr"
        self.assertEqual(document.metadata["language"], "en")
        with self.assertRaises(TypeError):
            document.metadata["language"] = "de"  # type: ignore[index]

    def test_empty_identifiers_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Document(" ", "text")
        with self.assertRaises(ValueError):
            Chunk("id", "", "text")

    def test_query_and_rank_validation(self) -> None:
        with self.assertRaises(ValueError):
            Query("  ")
        chunk = Chunk("c", "d", "text")
        with self.assertRaises(ValueError):
            SearchResult(chunk, 1.0, 0, "test")


if __name__ == "__main__":
    unittest.main()

