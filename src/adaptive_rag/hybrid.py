"""Unified high-level hybrid retrieval facade combining lexical, dense, and rank-fused search."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .cache import CachedRetriever
from .chunking import TokenChunker
from .composition import ReciprocalRankFusionRetriever
from .filtering import MetadataFilter
from .models import Chunk, Document, Query, SearchResult
from .retrievers.base import Retriever
from .retrievers.bm25 import BM25Retriever
from .retrievers.dense import DenseRetriever, Embedder
from .tokenization import NormalizedTokenizer


class HybridRetriever:
    """Convenience facade combining BM25, Dense, and rank-fused hybrid retrieval."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        split_identifiers: bool = True,
        anchor_boost: float = 1.0,
        anchor_gap_threshold: float = 0.25,
        weights: Sequence[float] = (1.0, 1.0),
        rank_constant: int = 60,
        candidate_multiplier: int = 3,
        revision: Callable[[], str] | None = None,
        scope: str = "default",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.tokenizer = NormalizedTokenizer(split_identifiers=split_identifiers)
        self.sparse = BM25Retriever(k1=k1, b=b, tokenizer=self.tokenizer)
        self.dense = DenseRetriever(embedder)
        self.fusion = ReciprocalRankFusionRetriever(
            [self.sparse, self.dense],
            weights=weights,
            rank_constant=rank_constant,
            candidate_multiplier=candidate_multiplier,
            anchor_boost=anchor_boost,
            anchor_gap_threshold=anchor_gap_threshold,
        )
        if revision is not None:
            self._backend: Retriever = CachedRetriever(
                self.fusion, revision=revision, scope=scope
            )
        else:
            self._backend = self.fusion

    def __len__(self) -> int:
        return len(self.sparse)

    def add(self, chunks: Iterable[Chunk]) -> None:
        """Add chunks directly to both sparse and dense indexes."""
        chunk_list = list(chunks)
        self.sparse.add(chunk_list)
        self.dense.add(chunk_list)

    def add_documents(
        self,
        documents: Iterable[Document],
        *,
        chunker: TokenChunker | None = None,
    ) -> list[Chunk]:
        """Chunk and index documents into both sparse and dense backends."""
        active_chunker = chunker or TokenChunker()
        chunks = active_chunker.chunk_many(documents)
        self.add(chunks)
        return chunks

    def add_text(
        self,
        text: str,
        *,
        document_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        chunker: TokenChunker | None = None,
    ) -> list[Chunk]:
        """Convenience method to chunk and index raw text directly."""
        doc_id = document_id or f"doc-{len(self.sparse) + 1:06d}"
        doc = Document(id=doc_id, text=text, metadata=metadata or {})
        return self.add_documents([doc], chunker=chunker)

    def add_file(
        self,
        file_path: str | Path,
        *,
        chunker: TokenChunker | None = None,
        **loader_kwargs: Any,
    ) -> list[Chunk]:
        """Load a file (text, markdown, json, pdf, image), chunk it, and index it."""
        from .loaders import load_file

        docs = load_file(file_path, **loader_kwargs)
        return self.add_documents(docs, chunker=chunker)

    def add_directory(
        self,
        directory_path: str | Path,
        *,
        glob: str = "**/*",
        chunker: TokenChunker | None = None,
        **loader_kwargs: Any,
    ) -> list[Chunk]:
        """Recursively load all supported files in a directory, chunk, and index."""
        from .loaders import DirectoryLoader

        docs = DirectoryLoader(directory_path, glob=glob, loader_kwargs=loader_kwargs).load()
        return self.add_documents(docs, chunker=chunker)

    def search(
        self,
        query: str | Query,
        *,
        top_k: int = 5,
        where: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """Search across both lexical and dense indexes with rank fusion."""
        return self._backend.search(query, top_k=top_k)
