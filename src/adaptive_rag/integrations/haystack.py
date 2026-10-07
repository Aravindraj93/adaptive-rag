"""Haystack (2.x) integration adapter for adaptive-rag.

Provides an AdaptiveHaystackRetriever component that can be used directly
in Haystack 2.x Pipelines with zero API cost and CPU-first execution.

Usage:
    from adaptive_rag.integrations.haystack import AdaptiveHaystackRetriever

    retriever = AdaptiveHaystackRetriever.from_texts(
        ["Refunds are available within thirty days.", "Shipping takes 3-5 days."],
        scope="haystack-app",
        top_k=3,
    )

    # In a Haystack 2.x pipeline:
    # pipeline.add_component("retriever", retriever)
    # result = pipeline.run({"retriever": {"query": "refund policy"}})

    # Direct component invocation:
    result = retriever.run(query="refund policy", top_k=2)
    documents = result["documents"]
"""

from __future__ import annotations

from typing import Any, Optional, Sequence


def _to_haystack_document(search_result: Any) -> Any:
    """Convert an adaptive-rag SearchResult to a Haystack Document."""
    chunk = search_result.chunk
    metadata = dict(chunk.metadata or {})
    source = metadata.get("source", chunk.document_id)

    try:
        from haystack.dataclasses import Document
        return Document(
            id=chunk.id,
            content=chunk.text,
            meta={
                "document_id": chunk.document_id,
                "source": source,
                **metadata,
            },
            score=getattr(search_result, "score", None),
        )
    except ImportError:
        # Fallback shim matching Haystack Document duck-typing
        class _SimpleHaystackDocument:
            def __init__(self, id: str, content: str, meta: dict, score: float | None) -> None:
                self.id = id
                self.content = content
                self.meta = meta
                self.score = score

            def to_dict(self) -> dict[str, Any]:
                return {"id": self.id, "content": self.content, "meta": self.meta, "score": self.score}

            def __repr__(self) -> str:
                return f"Document(id={self.id!r}, content={self.content[:40]!r}, score={self.score})"

        return _SimpleHaystackDocument(
            id=chunk.id,
            content=chunk.text,
            meta={
                "document_id": chunk.document_id,
                "source": source,
                **metadata,
            },
            score=getattr(search_result, "score", None),
        )


class AdaptiveHaystackRetriever:
    """Haystack 2.x compatible component backed by adaptive-rag.

    Can be placed inside Haystack Pipelines or called directly via .run().
    """

    def __init__(
        self,
        retriever: Any,
        *,
        top_k: int = 5,
        scope: str = "haystack-adaptive",
    ) -> None:
        self.retriever = retriever
        self.top_k = top_k
        self.scope = scope

    @classmethod
    def from_texts(
        cls,
        texts: Sequence[str],
        *,
        metadatas: Optional[Sequence[dict]] = None,
        scope: str = "haystack-adaptive",
        top_k: int = 5,
        use_cache: bool = True,
        revision: int = 1,
    ) -> "AdaptiveHaystackRetriever":
        """Build an AdaptiveHaystackRetriever from raw text strings."""
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

    def run(self, query: str, top_k: Optional[int] = None) -> dict[str, list[Any]]:
        """Haystack 2.x component entry point.

        Args:
            query: The search query string.
            top_k: Optional override for the number of results.

        Returns:
            Dictionary with a "documents" list of Haystack Document objects.
        """
        k = top_k if top_k is not None else self.top_k
        results = self.retriever.search(query, top_k=k)
        docs = [_to_haystack_document(r) for r in results]
        return {"documents": docs}

    def search(self, query: str, top_k: Optional[int] = None) -> list[Any]:
        """Convenience method returning adaptive-rag SearchResult objects."""
        return self.retriever.search(query, top_k=top_k or self.top_k)

    def __repr__(self) -> str:
        return f"AdaptiveHaystackRetriever(scope={self.scope!r}, top_k={self.top_k})"
