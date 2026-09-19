"""Unit tests for document, PDF, and image loaders."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from adaptive_rag import (
    DirectoryLoader,
    Document,
    HashingEmbedder,
    HybridRetriever,
    ImageLoader,
    JSONLoader,
    PDFLoader,
    TextLoader,
    load_file,
)


class TestLoaders(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_text_loader(self) -> None:
        file_path = self.base_path / "sample.txt"
        file_path.write_text("Line 1\nLine 2: REQ_001 drive controller.", encoding="utf-8")

        loader = TextLoader(file_path, metadata={"project": "Automotive"})
        docs = loader.load()

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].id, "sample.txt")
        self.assertIn("REQ_001", docs[0].text)
        self.assertEqual(docs[0].metadata["project"], "Automotive")
        self.assertEqual(docs[0].metadata["extension"], ".txt")

    def test_text_loader_missing_file(self) -> None:
        loader = TextLoader(self.base_path / "missing.txt")
        with self.assertRaises(FileNotFoundError):
            loader.load()

    def test_json_loader_array(self) -> None:
        file_path = self.base_path / "data.json"
        data = [
            {"id": "doc-1", "text": "First chunk text", "section": "intro"},
            {"id": "doc-2", "text": "Second chunk text", "section": "body"},
        ]
        file_path.write_text(json.dumps(data), encoding="utf-8")

        loader = JSONLoader(file_path)
        docs = loader.load()

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].id, "doc-1")
        self.assertEqual(docs[0].text, "First chunk text")
        self.assertEqual(docs[0].metadata["section"], "intro")
        self.assertEqual(docs[1].id, "doc-2")

    def test_json_loader_jsonl(self) -> None:
        file_path = self.base_path / "records.jsonl"
        lines = [
            json.dumps({"id": "rec-1", "text": "Log entry 1"}),
            json.dumps({"id": "rec-2", "text": "Log entry 2"}),
        ]
        file_path.write_text("\n".join(lines), encoding="utf-8")

        loader = JSONLoader(file_path)
        docs = loader.load()

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].id, "rec-1")
        self.assertEqual(docs[1].id, "rec-2")

    def test_pdf_loader_missing_pypdf_raises(self) -> None:
        file_path = self.base_path / "dummy.pdf"
        file_path.write_bytes(b"%PDF-1.4 dummy")

        loader = PDFLoader(file_path)
        with patch.dict("sys.modules", {"pypdf": None}):
            with self.assertRaises(ImportError) as ctx:
                loader.load()
            self.assertIn("pypdf is required for PDFLoader", str(ctx.exception))

    def test_pdf_loader_with_mocked_pypdf(self) -> None:
        file_path = self.base_path / "manual.pdf"
        file_path.write_bytes(b"%PDF-1.4 test")

        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1: System Overview and Architecture"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2: Electrical Schematics for POPUP_PU1436"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page1, mock_page2]
        mock_reader.is_encrypted = False

        mock_pypdf = MagicMock()
        mock_pypdf.PdfReader.return_value = mock_reader

        with patch.dict("sys.modules", {"pypdf": mock_pypdf}):
            # 1. Multi-page mode
            loader = PDFLoader(file_path, extract_pages=True)
            docs = loader.load()

            self.assertEqual(len(docs), 2)
            self.assertEqual(docs[0].id, "manual.pdf::page-1")
            self.assertEqual(docs[0].metadata["page"], 1)
            self.assertEqual(docs[0].metadata["total_pages"], 2)
            self.assertIn("System Overview", docs[0].text)

            self.assertEqual(docs[1].id, "manual.pdf::page-2")
            self.assertIn("POPUP_PU1436", docs[1].text)

            # 2. Single document mode
            single_loader = PDFLoader(file_path, extract_pages=False)
            single_docs = single_loader.load()
            self.assertEqual(len(single_docs), 1)
            self.assertIn("--- Page 1 ---", single_docs[0].text)
            self.assertIn("--- Page 2 ---", single_docs[0].text)

    def test_image_loader_with_vision_callback(self) -> None:
        file_path = self.base_path / "circuit_diagram.png"
        file_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        def mock_vision(img_path):
            return f"Diagram showing relay K1 and sensor S2 connected to DriveMode ECU at {img_path.name}."

        loader = ImageLoader(file_path, vision_fn=mock_vision)
        docs = loader.load()

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].id, "circuit_diagram.png")
        self.assertIn("relay K1", docs[0].text)
        self.assertEqual(docs[0].metadata["extractor"], "vision_fn")

    def test_image_loader_with_ocr_callback(self) -> None:
        file_path = self.base_path / "warning_label.jpg"
        file_path.write_bytes(b"\xff\xd8\xff")

        loader = ImageLoader(file_path, ocr_fn=lambda p: "DANGER: High Voltage 400V")
        docs = loader.load()

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].text, "DANGER: High Voltage 400V")
        self.assertEqual(docs[0].metadata["extractor"], "ocr_fn")

    def test_image_loader_missing_pillow_raises(self) -> None:
        file_path = self.base_path / "label.png"
        file_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        loader = ImageLoader(file_path)
        with patch.dict("sys.modules", {"PIL": None, "pytesseract": None}):
            with self.assertRaises(ImportError) as ctx:
                loader.load()
            self.assertIn("Pillow and pytesseract are required", str(ctx.exception))

    def test_directory_loader_and_load_file(self) -> None:
        (self.base_path / "sub").mkdir()
        (self.base_path / "a.txt").write_text("Document A text", encoding="utf-8")
        (self.base_path / "sub" / "b.md").write_text("# Document B Markdown", encoding="utf-8")
        (self.base_path / ".hidden.txt").write_text("Ignored hidden text", encoding="utf-8")

        # Test load_file
        doc_a = load_file(self.base_path / "a.txt")
        self.assertEqual(len(doc_a), 1)
        self.assertEqual(doc_a[0].text, "Document A text")

        # Test DirectoryLoader
        dir_loader = DirectoryLoader(self.base_path)
        all_docs = dir_loader.load()

        doc_ids = {d.id for d in all_docs}
        self.assertIn("a.txt", doc_ids)
        self.assertIn("b.md", doc_ids)
        self.assertNotIn(".hidden.txt", doc_ids)

    def test_hybrid_retriever_ingestion_api(self) -> None:
        embedder = HashingEmbedder(dimensions=16)
        retriever = HybridRetriever(embedder=embedder)

        # 1. add_text
        chunks1 = retriever.add_text(
            "REQ_CFTS081: The vehicle speed controller regulates throttle during DriveMode transitions.",
            document_id="spec-throttle",
        )
        self.assertTrue(len(chunks1) >= 1)

        # 2. add_file
        file_path = self.base_path / "safety_note.txt"
        file_path.write_text(
            "POPUP_PU1436: Overheat warning popup must trigger when inverter temp exceeds 85C.",
            encoding="utf-8",
        )
        chunks2 = retriever.add_file(file_path)
        self.assertTrue(len(chunks2) >= 1)

        self.assertEqual(len(retriever), len(chunks1) + len(chunks2))

        # 3. Search lexical and semantic queries
        res_req = retriever.search("REQ_CFTS081", top_k=2)
        self.assertTrue(len(res_req) >= 1)
        self.assertIn("DriveMode", res_req[0].chunk.text)

        res_popup = retriever.search("POPUP_PU1436 inverter overheat", top_k=2)
        self.assertTrue(len(res_popup) >= 1)
        self.assertIn("85C", res_popup[0].chunk.text)


if __name__ == "__main__":
    unittest.main()
