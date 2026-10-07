"""LlamaIndex integration adapter for adaptive-rag.

Provides a drop-in retriever and query engine that plugs adaptive-rag's
CPU-first retrieval into LlamaIndex pipelines.

Usage:
    from adaptive_rag.integrations.llamaindex import AdaptiveQueryEngine

    engine = AdaptiveQueryEngine.from_documents(documents, scope="my-app")
    response = engine.query("What is the refund policy?")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:
    pass  # llama_index imports are lazy


def _to_llama_node(search_result: Any) -> Any:
    """Convert adaptive-rag SearchResult to a LlamaIndex NodeWithScore."""
    chunk = search_result.chunk
    metadata = dict(chunk.metadata or {})
    source = metadata.get("source", chunk.document_id)
    try:
        from llama_index.core.schema import NodeWithScore, TextNode
        node = TextNode(
            text=chunk.text,
            id_=chunk.id,
            metadata={
                "document_id": chunk.document_id,
                "source": source,
                **metadata,
            },
        )
        return NodeWithScore(
            node=node,
            score=getattr(search_result, "score", 1.0),
        )
    except ImportError:
        return {
            "text": chunk.text,
            "id": chunk.id,
            "document_id": chunk.document_id,
            "score": getattr(search_result, "score", 1.0),
        }


def _from_llama_document(doc: Any) -> Any:
    """Convert a LlamaIndex Document to adaptive-rag Chunk."""
    from adaptive_rag import Chunk
    doc_id = getattr(doc, "doc_id", None) or getattr(doc, "id_", None) or str(id(doc))
    text = getattr(doc, "text", "") or getattr(doc, "get_content", lambda: "")()
    metadata = getattr(doc, "metadata", {}) or {}
    source = metadata.get("file_path") or metadata.get("source") or str(doc_id)
    return Chunk(
        id=str(doc_id),
        document_id=str(source),
        text=str(text),
        metadata=dict(metadata),
    )


class AdaptiveBaseRetriever:
    """LlamaIndex-compatible retriever backed by adaptive-rag.

    Implements the LlamaIndex BaseRetriever protocol using duck-typing
    so it works whether or not llama_index is installed.
    """

    def __init__(self, adaptive_retriever: Any, *, top_k: int = 5) -> None:
        self._retriever = adaptive_retriever
        self._top_k = top_k

    def retrieve(self, query: str, **kwargs: Any) -> list:
        """Retrieve nodes for a query (LlamaIndex protocol)."""
        results = self._retriever.search(query, top_k=self._top_k)
        return [_to_llama_node(r) for r in results]

    async def aretrieve(self, query: str, **kwargs: Any) -> list:
        """Async retrieval (runs sync, CPU is fast enough)."""
        return self.retrieve(query, **kwargs)

    def _retrieve(self, query: Any, **kwargs: Any) -> list:
        """Internal LlamaIndex protocol method."""
        query_str = getattr(query, "query_str", None) or str(query)
        return self.retrieve(query_str, **kwargs)


class AdaptiveQueryEngine:
    """High-level query engine bridging adaptive-rag and LlamaIndex.

    Features:
    - CPU-only retrieval — no GPU, no embedding API calls
    - Semantic caching built-in (reduces repeated query cost to ~0)
    - Works as a drop-in LlamaIndex query engine
    - Optional LLM synthesis (pass your own LLM)

    Examples:
        # From LlamaIndex Documents
        engine = AdaptiveQueryEngine.from_documents(docs, scope="my-app")
        response = engine.query("What is the return policy?")

        # From an existing LlamaIndex index (wraps its retriever)
        engine = AdaptiveQueryEngine.from_index(index, scope="my-app")

        # With a custom LLM for answer synthesis
        engine = AdaptiveQueryEngine.from_documents(docs, llm=my_llm)
    """

    def __init__(
        self,
        retriever: Any,
        *,
        llm: Optional[Any] = None,
        top_k: int = 5,
        scope: str = "adaptive-rag",
        synthesize: bool = True,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._top_k = top_k
        self._scope = scope
        self._synthesize = synthesize and llm is not None

    # ── Factory constructors ──────────────────────────────────────────────────

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[Any],
        *,
        scope: str = "adaptive-rag",
        top_k: int = 5,
        use_cache: bool = True,
        revision: int = 1,
        llm: Optional[Any] = None,
    ) -> "AdaptiveQueryEngine":
        """Build an engine from LlamaIndex Document objects.

        Args:
            documents: List of LlamaIndex Document objects.
            scope: Cache scope identifier (e.g. your app name).
            top_k: Number of results to retrieve per query.
            use_cache: Whether to wrap with SemanticCachedRetriever.
            revision: Cache revision; increment to invalidate.
            llm: Optional LLM for answer synthesis.

        Returns:
            AdaptiveQueryEngine ready to use.

        Example:
            from llama_index.core import SimpleDirectoryReader
            docs = SimpleDirectoryReader("./data").load_data()
            engine = AdaptiveQueryEngine.from_documents(docs, scope="my-docs")
            response = engine.query("What are the pricing tiers?")
        """
        from adaptive_rag import BM25Retriever, SemanticCachedRetriever

        chunks = [_from_llama_document(doc) for doc in documents]
        backend = BM25Retriever()
        backend.add(chunks)

        if use_cache:
            _rev = revision
            retriever = SemanticCachedRetriever(
                backend, scope=scope, revision=lambda: _rev
            )
        else:
            retriever = backend

        adaptive_retriever = AdaptiveBaseRetriever(retriever, top_k=top_k)
        return cls(adaptive_retriever, llm=llm, top_k=top_k, scope=scope)

    @classmethod
    def from_index(
        cls,
        index: Any,
        *,
        scope: str = "adaptive-rag",
        top_k: int = 5,
        use_cache: bool = True,
        revision: int = 1,
        llm: Optional[Any] = None,
    ) -> "AdaptiveQueryEngine":
        """Build an engine by extracting documents from a LlamaIndex index.

        Args:
            index: Any LlamaIndex index (VectorStoreIndex, SummaryIndex, etc.)
            scope: Cache scope identifier.
            top_k: Number of results to retrieve per query.
            use_cache: Whether to enable semantic caching.
            revision: Cache revision number.
            llm: Optional LLM for answer synthesis.

        Returns:
            AdaptiveQueryEngine ready to use.

        Example:
            from llama_index.core import VectorStoreIndex
            index = VectorStoreIndex.from_documents(documents)
            # Switch to adaptive-rag (CPU, no API cost):
            engine = AdaptiveQueryEngine.from_index(index, scope="my-index")
        """
        # Extract documents from the index's docstore
        try:
            docstore = index.docstore
            docs = list(docstore.docs.values())
        except AttributeError:
            docs = []

        if not docs:
            raise ValueError(
                "Could not extract documents from the provided index. "
                "Use AdaptiveQueryEngine.from_documents() instead."
            )

        return cls.from_documents(
            docs, scope=scope, top_k=top_k, use_cache=use_cache,
            revision=revision, llm=llm,
        )

    @classmethod
    def from_texts(
        cls,
        texts: Sequence[str],
        *,
        scope: str = "adaptive-rag",
        top_k: int = 5,
        use_cache: bool = True,
        revision: int = 1,
        llm: Optional[Any] = None,
    ) -> "AdaptiveQueryEngine":
        """Build an engine from plain text strings.

        Args:
            texts: List of text strings.
            scope: Cache scope identifier.
            top_k: Number of results per query.
            use_cache: Whether to enable semantic caching.
            revision: Cache revision number.
            llm: Optional LLM for answer synthesis.

        Returns:
            AdaptiveQueryEngine ready to use.
        """
        from adaptive_rag import BM25Retriever, Chunk, SemanticCachedRetriever

        chunks = [
            Chunk(id=f"text-{i}", document_id="text", text=t)
            for i, t in enumerate(texts)
        ]
        backend = BM25Retriever()
        backend.add(chunks)

        if use_cache:
            _rev = revision
            retriever = SemanticCachedRetriever(
                backend, scope=scope, revision=lambda: _rev
            )
        else:
            retriever = backend

        adaptive_retriever = AdaptiveBaseRetriever(retriever, top_k=top_k)
        return cls(adaptive_retriever, llm=llm, top_k=top_k, scope=scope)

    # ── Core query ────────────────────────────────────────────────────────────

    def query(self, query_str: str, **kwargs: Any) -> Any:
        """Execute a query against the corpus.

        If an LLM was provided, synthesizes an answer from retrieved contexts.
        Otherwise returns a Response-like object with the source nodes.

        Args:
            query_str: Natural language question.

        Returns:
            Response object with .response (str) and .source_nodes (list).
        """
        nodes = self._retriever.retrieve(query_str)

        if self._synthesize and self._llm:
            return self._synthesize_answer(query_str, nodes)

        def _get_text(n: Any) -> str:
            if isinstance(n, dict):
                return str(n.get("text", ""))
            if hasattr(n, "get_content") and callable(n.get_content):
                return str(n.get_content())
            if hasattr(n, "node") and hasattr(n.node, "text"):
                return str(n.node.text)
            if hasattr(n, "text"):
                return str(n.text)
            return str(n)

        return _SimpleResponse(
            response="\n\n".join(_get_text(n) for n in nodes),
            source_nodes=nodes,
            query=query_str,
        )

    async def aquery(self, query_str: str, **kwargs: Any) -> Any:
        """Async query (runs sync internally — CPU is fast enough)."""
        return self.query(query_str, **kwargs)

    def _synthesize_answer(self, query: str, nodes: list) -> Any:
        """Synthesize an answer using the provided LLM."""
        context = "\n\n".join(
            getattr(getattr(n, "node", n), "text", str(n))
            for n in nodes
        )
        prompt = (
            f"Answer the following question based only on the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        try:
            response_text = self._llm.complete(prompt).text
        except AttributeError:
            response_text = str(self._llm.invoke(prompt))

        return _SimpleResponse(
            response=response_text,
            source_nodes=nodes,
            query=query,
        )

    # ── LlamaIndex compatibility ───────────────────────────────────────────────

    def as_query_engine(self) -> "AdaptiveQueryEngine":
        """Return self (already a query engine). For API symmetry."""
        return self

    def as_retriever(self, **kwargs: Any) -> AdaptiveBaseRetriever:
        """Return the underlying retriever (LlamaIndex protocol)."""
        return self._retriever

    def __repr__(self) -> str:
        return (
            f"AdaptiveQueryEngine("
            f"scope={self._scope!r}, "
            f"top_k={self._top_k}, "
            f"llm={type(self._llm).__name__ if self._llm else None})"
        )


class _SimpleResponse:
    """Minimal response object compatible with LlamaIndex Response."""

    def __init__(self, response: str, source_nodes: list, query: str) -> None:
        self.response = response
        self.source_nodes = source_nodes
        self.query = query
        self.metadata = {}

    def __str__(self) -> str:
        return self.response

    def __repr__(self) -> str:
        return f"Response(response={self.response[:100]!r}, source_nodes={len(self.source_nodes)})"

    def get_formatted_sources(self, length: int = 100) -> str:
        """Return a formatted string of source node excerpts."""
        parts = []
        for i, node in enumerate(self.source_nodes, 1):
            text = (
                getattr(getattr(node, "node", node), "text", "")
                or str(node)
            )
            parts.append(f"[{i}] {text[:length]}...")
        return "\n".join(parts)
