import unittest

from adaptive_rag import Document, TokenChunker


class ChunkingTests(unittest.TestCase):
    def test_overlapping_chunks_preserve_text_and_offsets(self) -> None:
        document = Document("doc", "one two three four five six seven", {"source": "test"})
        chunks = TokenChunker(chunk_size=3, overlap=1).chunk(document)
        self.assertEqual([chunk.text for chunk in chunks], [
            "one two three",
            "three four five",
            "five six seven",
        ])
        self.assertEqual([chunk.id for chunk in chunks], [
            "doc::chunk-000000",
            "doc::chunk-000001",
            "doc::chunk-000002",
        ])
        self.assertEqual(chunks[1].metadata["chunk_start_token"], 2)
        self.assertEqual(chunks[1].metadata["source"], "test")

    def test_empty_document_produces_no_chunks(self) -> None:
        self.assertEqual(TokenChunker().chunk(Document("empty", "  \n")), [])

    def test_invalid_overlap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TokenChunker(chunk_size=4, overlap=4)


if __name__ == "__main__":
    unittest.main()

