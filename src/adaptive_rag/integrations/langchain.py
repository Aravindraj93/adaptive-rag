"""LangChain integration adapter for adaptive-rag.

Provides a drop-in retriever that plugs adaptive-rag's CPU-first retrieval
into LangChain pipelines with zero API cost.

Usage:
    from adaptive_rag.integrations.langchain import AdaptiveRetriever

    retriever = AdaptiveRetriever.from_corpus(documents, scope="my-app")
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever.as_langchain_retriever())
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

if TYPE_CHECKING:
    # Only import when type-checking to avoid hard dependency
    from langchain_core.documents import Document as LangChainDocument
    from langchain_core.retrievers import BaseRetriever


class _LangChainRetrieverWrapper:
    """A thin shim that makes adaptive-rag look like a LangChain BaseRetriever.

    This class uses duck-typing so it works even if langchain is not installed.
    Install it with: pip install langchain-core
    """

    def __init__(self, adaptive_retriever: Any, top_k: int = 5) -> None:
        self._retriever = adaptive_retriever
        self._top_k = top_k

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list:
        """LangChain's synchronous retrieval protocol."""
        results = self._retriever.search(query, top_k=self._top_k)
        return [_to_langchain_document(r) for r in results]

    async def _aget_relevant_documents(self, query: str, *, run_manager: Any = None) -> list:
        """LangChain's async retrieval protocol."""
        # adaptive-rag search is fast enough on CPU; run sync
        return self._get_relevant_documents(query, run_manager=run_manager)

    def get_relevant_documents(self, query: str) -> list:
        """LangChain compatibility shim."""
        return self._get_relevant_documents(query)

    async def aget_relevant_documents(self, query: str) -> list:
        return await self._aget_relevant_documents(query)

    # Make it quack like a BaseRetriever for chain compatibility
    def __call__(self, query: str) -> list:
        return self.get_relevant_documents(query)


def _to_langchain_document(search_result: Any) -> Any:
    """Convert adaptive-rag SearchResult to a LangChain Document."""
    chunk = search_result.chunk
    metadata = dict(chunk.metadata or {})
    source = metadata.get("source", chunk.document_id)
    try:
        from langchain_core.documents import Document
        return Document(
            page_content=chunk.text,
            metadata={
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "source": source,
                "score": getattr(search_result, "score", None),
                **metadata,
            },
        )
    except ImportError:
        class _SimpleLangChainDoc:
            def __init__(self, page_content: str, metadata: dict) -> None:
                self.page_content = page_content
                self.metadata = metadata
            def __getitem__(self, item: str) -> Any:
                return getattr(self, item)
            def __repr__(self) -> str:
                return f"Document(page_content={self.page_content[:40]!r}, metadata={self.metadata})"

        return _SimpleLangChainDoc(
            page_content=chunk.text,
            metadata={
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "source": source,
                "score": getattr(search_result, "score", None),
                **metadata,
            },
        )


def _from_langchain_document(doc: Any) -> Any:
    """Convert a LangChain Document to adaptive-rag Chunk."""
    from adaptive_rag import Chunk
    chunk_id = (
        doc.metadata.get("chunk_id")
        or doc.metadata.get("id")
        or str(hash(doc.page_content))[:12]
    )
    doc_id = (
        doc.metadata.get("document_id")
        or doc.metadata.get("source")
        or doc.metadata.get("file_path")
        or "langchain"
    )
    metadata = {k: v for k, v in doc.metadata.items() if k not in ("chunk_id",)}
    return Chunk(
        id=str(chunk_id),
        document_id=str(doc_id),
        text=doc.page_content,
        metadata=metadata if metadata else {},
    )


class AdaptiveRetriever:
    """High-level adapter that bridges adaptive-rag and LangChain.

    Features:
    - Zero mandatory dependencies (langchain optional)
    - CPU-only retrieval — no GPU, no embedding API calls
    - Semantic caching built-in
    - Works as a drop-in LangChain BaseRetriever

    Examples:
        # From a list of LangChain documents
        retriever = AdaptiveRetriever.from_langchain_docs(docs, scope="my-app")

        # From raw texts
        retriever = AdaptiveRetriever.from_texts(texts, scope="my-app")

        # As a LangChain retriever
        lc_retriever = retriever.as_langchain_retriever()
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=lc_retriever)
    """

    def __init__(
        self,
        retriever: Any,
        *,
        top_k: int = 5,
        scope: str = "default",
    ) -> None:
        self._retriever = retriever
        self._top_k = top_k
        self._scope = scope

    # ── Factory constructors ──────────────────────────────────────────────────

    @classmethod
    def from_langchain_docs(
        cls,
        documents: Sequence[Any],
        *,
        scope: str = "adaptive-rag",
        top_k: int = 5,
        use_cache: bool = True,
        revision: int = 1,
    ) -> "AdaptiveRetriever":
        """Build an AdaptiveRetriever from LangChain Document objects.

        Args:
            documents: List of LangChain Document objects.
            scope: Cache scope identifier (e.g. your app name).
            top_k: Default number of results to return.
            use_cache: Whether to wrap with SemanticCachedRetriever.
            revision: Cache revision; increment to invalidate.

        Returns:
            AdaptiveRetriever ready to use.
        """
        from adaptive_rag import BM25Retriever, SemanticCachedRetriever

        chunks = [_from_langchain_document(doc) for doc in documents]
        backend = BM25Retriever()
        backend.add(chunks)

        if use_cache:
            _rev = revision
            retriever = SemanticCachedRetriever(
                backend,
                scope=scope,
                revision=lambda: _rev,
            )
        else:
            retriever = backend

        return cls(retriever, top_k=top_k, scope=scope)

    @classmethod
    def from_texts(
        cls,
        texts: Sequence[str],
        *,
        metadatas: Optional[Sequence[dict]] = None,
        scope: str = "adaptive-rag",
        top_k: int = 5,
        use_cache: bool = True,
        revision: int = 1,
    ) -> "AdaptiveRetriever":
        """Build an AdaptiveRetriever from plain text strings.

        Args:
            texts: List of text strings.
            metadatas: Optional list of metadata dicts (same length as texts).
            scope: Cache scope identifier.
            top_k: Default number of results to return.
            use_cache: Whether to enable semantic caching.
            revision: Cache revision number.

        Returns:
            AdaptiveRetriever ready to use.
        """
        from adaptive_rag import BM25Retriever, Chunk, SemanticCachedRetriever

        metas = metadatas or [{}] * len(texts)
        chunks = [
            Chunk(
                id=f"text-{i}",
                document_id=str(m.get("source", m.get("document_id", "text"))),
                text=t,
                metadata=m if m else {},
            )
            for i, (t, m) in enumerate(zip(texts, metas))
        ]

        backend = BM25Retriever()
        backend.add(chunks)

        if use_cache:
            _rev = revision
            retriever = SemanticCachedRetriever(
                backend,
                scope=scope,
                revision=lambda: _rev,
            )
        else:
            retriever = backend

        return cls(retriever, top_k=top_k, scope=scope)

    @classmethod
    def from_existing_retriever(
        cls,
        adaptive_retriever: Any,
        *,
        top_k: int = 5,
        scope: str = "adaptive-rag",
    ) -> "AdaptiveRetriever":
        """Wrap any existing adaptive-rag retriever.

        Args:
            adaptive_retriever: Any adaptive_rag retriever (BM25Retriever,
                HybridRetriever, SemanticCachedRetriever, etc.)
            top_k: Default number of results to return.
            scope: Label for this retriever instance.

        Returns:
            AdaptiveRetriever ready to use.
        """
        return cls(adaptive_retriever, top_k=top_k, scope=scope)

    # ── Core search ───────────────────────────────────────────────────────────

    def search(self, query: str, *, top_k: Optional[int] = None) -> list:
        """Search the corpus.

        Args:
            query: Natural language query string.
            top_k: Number of results (overrides instance default).

        Returns:
            List of adaptive_rag.SearchResult objects.
        """
        return self._retriever.search(query, top_k=top_k or self._top_k)

    def get_relevant_documents(self, query: str) -> list:
        """Return LangChain Documents for a query (LangChain protocol)."""
        results = self.search(query)
        return [_to_langchain_document(r) for r in results]

    # ── LangChain compatibility ───────────────────────────────────────────────

    def as_langchain_retriever(self, *, top_k: Optional[int] = None) -> Any:
        """Return an object that satisfies LangChain's BaseRetriever protocol.

        Use this when passing to LangChain chains, agents, or pipelines.

        Args:
            top_k: Override the number of results (optional).

        Returns:
            A LangChain-compatible retriever object.

        Example:
            retriever = AdaptiveRetriever.from_texts(texts, scope="my-app")
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever.as_langchain_retriever(),
            )
        """
        try:
            # Try to return a proper LangChain BaseRetriever subclass
            from langchain_core.retrievers import BaseRetriever
            from langchain_core.callbacks import CallbackManagerForRetrieverRun
            from langchain_core.documents import Document

            adaptive_self = self
            k = top_k or self._top_k

            class _ProperLangChainRetriever(BaseRetriever):
                def _get_relevant_documents(
                    self,
                    query: str,
                    *,
                    run_manager: CallbackManagerForRetrieverRun,
                ) -> list[Document]:
                    return adaptive_self.get_relevant_documents(query)

                async def _aget_relevant_documents(
                    self,
                    query: str,
                    *,
                    run_manager: CallbackManagerForRetrieverRun,
                ) -> list[Document]:
                    return adaptive_self.get_relevant_documents(query)

            return _ProperLangChainRetriever()

        except ImportError:
            # langchain not installed — return duck-typed wrapper
            return _LangChainRetrieverWrapper(self._retriever, top_k=top_k or self._top_k)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def add_documents(self, documents: Sequence[Any]) -> None:
        """Add LangChain Documents to the index (if backend supports it)."""
        from adaptive_rag import Chunk
        chunks = [_from_langchain_document(doc) for doc in documents]
        if hasattr(self._retriever, "add"):
            self._retriever.add(chunks)
        else:
            raise TypeError(
                f"Retriever {type(self._retriever).__name__} does not support .add(). "
                "Use a mutable retriever like BM25Retriever."
            )

    def __repr__(self) -> str:
        return (
            f"AdaptiveRetriever("
            f"retriever={type(self._retriever).__name__}, "
            f"scope={self._scope!r}, "
            f"top_k={self._top_k})"
        )
