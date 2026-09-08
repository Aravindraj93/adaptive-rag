"""Deterministic, dependency-free document chunking policies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .models import Chunk, Document

_TOKEN_WITH_SPAN = re.compile(r"(?u)\S+")


@dataclass(frozen=True, slots=True)
class TokenChunker:
    """Split documents into overlapping token windows.

    Whitespace-delimited spans are used so the original text, punctuation, and
    casing are preserved. Chunk ids and offsets are stable for identical input.
    """

    chunk_size: int = 256
    overlap: int = 32

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")

    def chunk(self, document: Document) -> list[Chunk]:
        spans = list(_TOKEN_WITH_SPAN.finditer(document.text))
        if not spans:
            return []

        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        for position, start_token in enumerate(range(0, len(spans), step)):
            end_token = min(start_token + self.chunk_size, len(spans))
            start_char = spans[start_token].start()
            end_char = spans[end_token - 1].end()
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "chunk_start_token": start_token,
                    "chunk_end_token": end_token,
                    "chunk_start_char": start_char,
                    "chunk_end_char": end_char,
                }
            )
            chunks.append(
                Chunk(
                    id=f"{document.id}::chunk-{position:06d}",
                    document_id=document.id,
                    text=document.text[start_char:end_char],
                    metadata=metadata,
                    position=position,
                )
            )
            if end_token == len(spans):
                break
        return chunks

    def chunk_many(self, documents: Iterable[Document]) -> list[Chunk]:
        return [chunk for document in documents for chunk in self.chunk(document)]

