import json
import tempfile
import unittest
from pathlib import Path

from adaptive_rag import BM25Retriever, Chunk, IndexFormatError, NormalizedTokenizer


class PersistenceTests(unittest.TestCase):
    def test_round_trip_preserves_ranking_and_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            original = BM25Retriever(
                k1=1.2,
                b=0.6,
                tokenizer=NormalizedTokenizer(fold_accents=True),
            )
            original.add([
                Chunk("a", "d", "Café products are returned quickly", {"kind": "policy"}),
                Chunk("b", "d", "Shipping guide and delivery schedule"),
            ])
            original.save(path)
            restored = BM25Retriever.load(path)
            self.assertEqual(restored.k1, 1.2)
            self.assertEqual(restored.b, 0.6)
            self.assertEqual(restored.search("cafe product")[0].chunk.id, "a")
            self.assertEqual(restored.search("cafe product")[0].chunk.metadata["kind"], "policy")

    def test_checksum_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            retriever = BM25Retriever()
            retriever.add([Chunk("a", "d", "original text")])
            retriever.save(path)
            envelope = json.loads(path.read_text(encoding="utf-8"))
            envelope["payload"]["chunks"][0]["text"] = "tampered text"
            path.write_text(json.dumps(envelope), encoding="utf-8")
            with self.assertRaisesRegex(IndexFormatError, "checksum"):
                BM25Retriever.load(path)

    def test_non_json_metadata_is_rejected_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            retriever = BM25Retriever()
            retriever.add([Chunk("a", "d", "text", {"bad": object()})])
            with self.assertRaises(TypeError):
                retriever.save(path)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()

