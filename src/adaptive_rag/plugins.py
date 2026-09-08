"""Domain-neutral extension protocols and a deterministic plugin pipeline."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol, Sequence

from .models import Document, Query


class DocumentEnricher(Protocol):
    """Add application-owned metadata without changing core model semantics."""

    name: str

    def enrich_document(self, document: Document) -> Mapping[str, Any]: ...


class QueryAnalyzer(Protocol):
    """Add application-owned query hints for downstream retrieval strategies."""

    name: str

    def analyze_query(self, query: Query) -> Mapping[str, Any]: ...


class PluginPipeline:
    """Apply ordered metadata extensions with explicit conflict detection."""

    def __init__(
        self,
        *,
        document_enrichers: Sequence[DocumentEnricher] = (),
        query_analyzers: Sequence[QueryAnalyzer] = (),
    ) -> None:
        self.document_enrichers = tuple(document_enrichers)
        self.query_analyzers = tuple(query_analyzers)

    def enrich_documents(self, documents: Iterable[Document]) -> list[Document]:
        enriched: list[Document] = []
        for document in documents:
            metadata = dict(document.metadata)
            for plugin in self.document_enrichers:
                self._merge(metadata, plugin.enrich_document(document), plugin.name)
            enriched.append(Document(document.id, document.text, metadata))
        return enriched

    def analyze_query(self, query: str | Query) -> Query:
        analyzed = query if isinstance(query, Query) else Query(query)
        metadata = dict(analyzed.metadata)
        for plugin in self.query_analyzers:
            self._merge(metadata, plugin.analyze_query(analyzed), plugin.name)
        return Query(analyzed.text, metadata)

    @staticmethod
    def _merge(target: dict[str, Any], additions: Mapping[str, Any], plugin_name: str) -> None:
        conflicts = target.keys() & additions.keys()
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(f"plugin {plugin_name!r} attempted to overwrite metadata: {names}")
        target.update(additions)

