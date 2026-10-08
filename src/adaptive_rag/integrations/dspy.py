"""DSPy integration adapter for adaptive-rag.

Provides AdaptiveDSPyRetriever that implements DSPy's dspy.Retrieve protocol.
Enables compiling and optimizing prompt pipelines with DSPy using fast, CPU-first,
revision-cached adaptive retrieval with zero API bills.

Usage:
    from adaptive_rag.integrations.dspy import AdaptiveDSPyRetriever

    retriever = AdaptiveDSPyRetriever.from_texts(
        ["Refund policy allows returns within 30 days.", "Support email is help@example.com."],
        k=3,
        scope="dspy-app",
    )

    # In DSPy:
    # dspy.settings.configure(rm=retriever)
    # retrieve = dspy.Retrieve(k=3)
    # passages = retrieve("refund policy").passages

    # Or direct call:
    passages = retriever("refund policy", k=2)
"""

from __future__ import annotations

from typing import Any, Optional, Sequence


def _to_dspy_passage(search_result: Any) -> Any:
    """Convert an adaptive-rag SearchResult to a DSPy compatible passage or dspy.Prediction."""
    chunk = search_result.chunk
    metadata = dict(chunk.metadata or {})
    source = metadata.get("source", chunk.document_id)

    try:
        import dspy
        return dspy.Prediction(
            long_text=chunk.text,
            id=chunk.id,
            document_id=chunk.document_id,
            source=source,
            score=getattr(search_result, "score", None),
        )
    except ImportError:
        # Fallback dictionary and object shim
        class _DSPyPassageShim:
            def __init__(self, long_text: str, id: str, document_id: str, source: str, score: float | None) -> None:
                self.long_text = long_text
                self.id = id
                self.document_id = document_id
                self.source = source
                self.score = score

            def __str__(self) -> str:
                return self.long_text

            def __repr__(self) -> str:
                return f"Passage(long_text={self.long_text[:40]!r}, score={self.score})"

        return _DSPyPassageShim(
            long_text=chunk.text,
            id=chunk.id,
            document_id=chunk.document_id,
            source=source,
            score=getattr(search_result, "score", None),
        )


class AdaptiveDSPyRetriever:
    """DSPy-compatible Retriever Model (RM) powered by adaptive-rag on CPU."""

    def __init__(
        self,
        retriever: Any,
        *,
        k: int = 5,
        scope: str = "dspy-adaptive",
    ) -> None:
        self.retriever = retriever
        self.k = k
        self.scope = scope

    @classmethod
    def from_texts(
        cls,
        texts: Sequence[str],
        *,
        metadatas: Optional[Sequence[dict]] = None,
        scope: str = "dspy-adaptive",
        k: int = 5,
        use_cache: bool = True,
        revision: int = 1,
    ) -> "AdaptiveDSPyRetriever":
        """Build an AdaptiveDSPyRetriever from plain text strings."""
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

        return cls(retriever, k=k, scope=scope)

    def forward(self, query_or_queries: str | list[str], k: Optional[int] = None, **kwargs: Any) -> list[Any]:
        """DSPy forward retrieval protocol.

        Args:
            query_or_queries: A single query string or list of queries.
            k: Optional override for the number of results.

        Returns:
            List of passages (or list of list of passages if multiple queries provided).
        """
        top_k = k if k is not None else self.k

        if isinstance(query_or_queries, str):
            results = self.retriever.search(query_or_queries, top_k=top_k)
            return [_to_dspy_passage(r) for r in results]

        # Multi-query handling
        all_results = []
        for q in query_or_queries:
            results = self.retriever.search(q, top_k=top_k)
            all_results.append([_to_dspy_passage(r) for r in results])
        return all_results

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allow calling retriever instance directly like a DSPy retriever."""
        return self.forward(*args, **kwargs)

    def search(self, query: str, top_k: Optional[int] = None) -> list[Any]:
        """Direct search returning adaptive-rag SearchResult objects."""
        return self.retriever.search(query, top_k=top_k or self.k)

    def __repr__(self) -> str:
        return f"AdaptiveDSPyRetriever(scope={self.scope!r}, k={self.k})"
