"""Document loaders for plain text, JSON, PDF, and image sources.

This module provides loaders that convert files into standard `Document`
instances ready for chunking and retrieval. The core library maintains zero
mandatory external dependencies; PDF and image OCR features require optional
extras (`pip install "adaptive-rag[pdf]"` or `"adaptive-rag[images]"`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from .models import Document


class BaseLoader:
    """Abstract interface for loading documents from file or data sources."""

    def load(self) -> list[Document]:
        """Load all documents synchronously."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """Yield documents one by one."""
        raise NotImplementedError


class TextLoader(BaseLoader):
    """Load plain text, markdown, code, or CSV files into Documents."""

    def __init__(
        self,
        file_path: str | Path,
        encoding: str = "utf-8",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.metadata = dict(metadata or {})

    def lazy_load(self) -> Iterator[Document]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        text = self.file_path.read_text(encoding=self.encoding)
        meta = {
            "source": str(self.file_path),
            "file_name": self.file_path.name,
            "extension": self.file_path.suffix.lower(),
            **self.metadata,
        }
        yield Document(id=self.file_path.name, text=text, metadata=meta)


class JSONLoader(BaseLoader):
    """Load JSON or JSONL files into Documents."""

    def __init__(
        self,
        file_path: str | Path,
        *,
        text_key: str = "text",
        id_key: str | None = "id",
        encoding: str = "utf-8",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.text_key = text_key
        self.id_key = id_key
        self.encoding = encoding
        self.metadata = dict(metadata or {})

    def lazy_load(self) -> Iterator[Document]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        suffix = self.file_path.suffix.lower()
        if suffix == ".jsonl":
            with self.file_path.open("r", encoding=self.encoding) as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        continue
                    text = str(record.get(self.text_key, ""))
                    doc_id = (
                        str(record.get(self.id_key, f"{self.file_path.name}::{idx}"))
                        if self.id_key
                        else f"{self.file_path.name}::{idx}"
                    )
                    meta = {
                        "source": str(self.file_path),
                        "line": idx,
                        **self.metadata,
                        **{k: v for k, v in record.items() if k != self.text_key},
                    }
                    yield Document(id=doc_id, text=text, metadata=meta)
        else:
            with self.file_path.open("r", encoding=self.encoding) as f:
                data = json.load(f)

            if isinstance(data, list):
                for idx, record in enumerate(data):
                    if isinstance(record, dict):
                        text = str(record.get(self.text_key, ""))
                        doc_id = (
                            str(record.get(self.id_key, f"{self.file_path.name}::{idx}"))
                            if self.id_key
                            else f"{self.file_path.name}::{idx}"
                        )
                        meta = {
                            "source": str(self.file_path),
                            "index": idx,
                            **self.metadata,
                            **{k: v for k, v in record.items() if k != self.text_key},
                        }
                        yield Document(id=doc_id, text=text, metadata=meta)
                    else:
                        yield Document(
                            id=f"{self.file_path.name}::{idx}",
                            text=str(record),
                            metadata={"source": str(self.file_path), "index": idx, **self.metadata},
                        )
            elif isinstance(data, dict):
                text = str(data.get(self.text_key, json.dumps(data)))
                doc_id = str(data.get(self.id_key, self.file_path.name)) if self.id_key else self.file_path.name
                meta = {
                    "source": str(self.file_path),
                    **self.metadata,
                    **{k: v for k, v in data.items() if k != self.text_key},
                }
                yield Document(id=doc_id, text=text, metadata=meta)


class PDFLoader(BaseLoader):
    """Extract text from PDF documents using `pypdf`.

    Requires `pypdf` optional dependency:
        pip install "adaptive-rag[pdf]"
    """

    def __init__(
        self,
        file_path: str | Path,
        *,
        extract_pages: bool = True,
        password: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.extract_pages = extract_pages
        self.password = password
        self.metadata = dict(metadata or {})

    def lazy_load(self) -> Iterator[Document]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "pypdf is required for PDFLoader. Install it with: "
                "pip install 'adaptive-rag[pdf]'"
            ) from exc

        reader = PdfReader(str(self.file_path))
        if self.password is not None and reader.is_encrypted:
            reader.decrypt(self.password)

        total_pages = len(reader.pages)
        if self.extract_pages:
            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                doc_id = f"{self.file_path.name}::page-{page_idx + 1}"
                meta = {
                    "source": str(self.file_path),
                    "file_name": self.file_path.name,
                    "page": page_idx + 1,
                    "total_pages": total_pages,
                    **self.metadata,
                }
                yield Document(id=doc_id, text=page_text, metadata=meta)
        else:
            full_text_parts = []
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                full_text_parts.append(f"--- Page {page_idx + 1} ---\n{text}")
            full_text = "\n\n".join(full_text_parts)
            meta = {
                "source": str(self.file_path),
                "file_name": self.file_path.name,
                "total_pages": total_pages,
                **self.metadata,
            }
            yield Document(id=self.file_path.name, text=full_text, metadata=meta)


class ImageLoader(BaseLoader):
    """Load text and descriptions from images using OCR or vision callbacks.

    Supports:
    1. Custom vision functions (`vision_fn(image_path)` -> text description),
       ideal for multimodal models (e.g. Gemini, GPT-4o, Claude Vision, local VLMs).
    2. Custom OCR functions (`ocr_fn(pil_image_or_path)` -> extracted text).
    3. Tesseract OCR via `pytesseract` and `Pillow`.

    Requires `pillow` and `pytesseract` optional dependencies for default OCR:
        pip install "adaptive-rag[images]"
    """

    def __init__(
        self,
        file_path: str | Path,
        *,
        vision_fn: Callable[[str | Path], str] | None = None,
        ocr_fn: Callable[[Any], str] | None = None,
        lang: str = "eng",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.vision_fn = vision_fn
        self.ocr_fn = ocr_fn
        self.lang = lang
        self.metadata = dict(metadata or {})

    def lazy_load(self) -> Iterator[Document]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        meta = {
            "source": str(self.file_path),
            "file_name": self.file_path.name,
            "extension": self.file_path.suffix.lower(),
            **self.metadata,
        }

        # 1. High priority: Custom vision callback (multimodal LLMs / descriptions)
        if self.vision_fn is not None:
            text = self.vision_fn(self.file_path)
            meta["extractor"] = "vision_fn"
            yield Document(id=self.file_path.name, text=str(text), metadata=meta)
            return

        # 2. Custom OCR callback
        if self.ocr_fn is not None:
            text = self.ocr_fn(self.file_path)
            meta["extractor"] = "ocr_fn"
            yield Document(id=self.file_path.name, text=str(text), metadata=meta)
            return

        # 3. Default: Tesseract OCR via Pillow
        try:
            from PIL import Image
            import pytesseract
        except ImportError as exc:
            raise ImportError(
                "Pillow and pytesseract are required for default ImageLoader OCR. "
                "Install them with: pip install 'adaptive-rag[images]' "
                "or supply a custom vision_fn or ocr_fn callback."
            ) from exc

        with Image.open(self.file_path) as img:
            extracted_text = pytesseract.image_to_string(img, lang=self.lang)

        meta["extractor"] = "tesseract_ocr"
        meta["ocr_lang"] = self.lang
        yield Document(id=self.file_path.name, text=extracted_text, metadata=meta)


_LOADER_REGISTRY: dict[str, type[BaseLoader]] = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".markdown": TextLoader,
    ".csv": TextLoader,
    ".log": TextLoader,
    ".py": TextLoader,
    ".c": TextLoader,
    ".cpp": TextLoader,
    ".h": TextLoader,
    ".java": TextLoader,
    ".js": TextLoader,
    ".ts": TextLoader,
    ".html": TextLoader,
    ".xml": TextLoader,
    ".yaml": TextLoader,
    ".yml": TextLoader,
    ".json": JSONLoader,
    ".jsonl": JSONLoader,
    ".pdf": PDFLoader,
    ".png": ImageLoader,
    ".jpg": ImageLoader,
    ".jpeg": ImageLoader,
    ".bmp": ImageLoader,
    ".webp": ImageLoader,
    ".tiff": ImageLoader,
}


def load_file(file_path: str | Path, **loader_kwargs: Any) -> list[Document]:
    """Auto-detect file format by extension and load documents."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    loader_cls = _LOADER_REGISTRY.get(suffix, TextLoader)
    loader = loader_cls(path, **loader_kwargs)
    return loader.load()


class DirectoryLoader(BaseLoader):
    """Recursively scan a directory and load documents across supported file types."""

    def __init__(
        self,
        directory_path: str | Path,
        *,
        glob: str = "**/*",
        recursive: bool = True,
        ignore_hidden: bool = True,
        loaders: Mapping[str, type[BaseLoader]] | None = None,
        loader_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.directory_path = Path(directory_path)
        self.glob = glob
        self.recursive = recursive
        self.ignore_hidden = ignore_hidden
        self.loaders = dict(loaders or {})
        self.loader_kwargs = dict(loader_kwargs or {})

    def lazy_load(self) -> Iterator[Document]:
        if not self.directory_path.exists() or not self.directory_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {self.directory_path}")

        files = (
            self.directory_path.glob(self.glob)
            if self.recursive
            else self.directory_path.glob(self.glob.replace("**/", ""))
        )

        for path in sorted(files):
            if not path.is_file():
                continue
            if self.ignore_hidden and any(part.startswith(".") for part in path.parts):
                continue

            suffix = path.suffix.lower()
            loader_cls = self.loaders.get(suffix) or _LOADER_REGISTRY.get(suffix)
            if loader_cls is None:
                continue

            try:
                loader = loader_cls(path, **self.loader_kwargs)
                yield from loader.lazy_load()
            except Exception:
                # Silently skip unreadable files in directory scan
                continue
