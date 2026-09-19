"""Domain-neutral extension protocols and a deterministic plugin pipeline."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .models import Chunk, Document, Query, SearchResult
from .retrievers.base import Retriever


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


class MetadataScoreModifier:
    """Adjust retrieval candidate scores using metadata match multipliers."""

    def __init__(
        self,
        retriever: Retriever,
        multipliers: Mapping[str, Mapping[str, float]],
        *,
        default_multiplier: float = 1.0,
        candidate_multiplier: int = 2,
    ) -> None:
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1")
        self.retriever = retriever
        self.multipliers = {
            field: dict(val_map) for field, val_map in multipliers.items()
        }
        self.default_multiplier = default_multiplier
        self.candidate_multiplier = candidate_multiplier

    def search(self, query: str | Query, *, top_k: int = 5) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        candidates = self.retriever.search(query, top_k=top_k * self.candidate_multiplier)
        rescored: list[tuple[float, SearchResult]] = []
        for cand in candidates:
            mult = 1.0
            matched = False
            for field, val_map in self.multipliers.items():
                if field in cand.chunk.metadata:
                    val_str = str(cand.chunk.metadata[field])
                    if val_str in val_map:
                        mult *= val_map[val_str]
                        matched = True
            if not matched:
                mult *= self.default_multiplier
            rescored.append((cand.score * mult, cand))

        rescored.sort(key=lambda item: (-item[0], item[1].chunk.id))
        return [
            SearchResult(
                cand.chunk,
                score,
                rank,
                f"metadata_modified:{cand.source}",
            )
            for rank, (score, cand) in enumerate(rescored[:top_k], start=1)
        ]


class FastPathIDLookupRetriever:
    """Fast-path resolver for exact identifier queries with fallback to general retrieval."""

    def __init__(
        self,
        fallback_retriever: Retriever,
        id_index: Mapping[str, Chunk],
        *,
        id_pattern: str | re.Pattern[str] = r"^[A-Za-z0-9]+(?:[_\-][A-Za-z0-9]+)+$",
        case_sensitive: bool = False,
    ) -> None:
        self.fallback_retriever = fallback_retriever
        self.case_sensitive = case_sensitive
        if case_sensitive:
            self.id_index = dict(id_index)
        else:
            self.id_index = {k.casefold(): v for k, v in id_index.items()}
        self.id_pattern = (
            re.compile(id_pattern) if isinstance(id_pattern, str) else id_pattern
        )

    def search(self, query: str | Query, *, top_k: int = 5) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_text = query.text if isinstance(query, Query) else query
        stripped = query_text.strip()
        matched_chunk: Chunk | None = None

        if self.id_pattern.match(stripped):
            key = stripped if self.case_sensitive else stripped.casefold()
            matched_chunk = self.id_index.get(key)
            if not matched_chunk and "_" in key:
                prefix, remainder = key.split("_", 1)
                if prefix in ("req", "popup", "sig"):
                    matched_chunk = self.id_index.get(remainder)

        if matched_chunk is not None:
            results: list[SearchResult] = [
                SearchResult(matched_chunk, 1.0, 1, "fastpath:exact_id")
            ]
            if top_k > 1:
                fallback_results = self.fallback_retriever.search(query, top_k=top_k)
                for fb in fallback_results:
                    if fb.chunk.id != matched_chunk.id and len(results) < top_k:
                        results.append(
                            SearchResult(
                                fb.chunk,
                                fb.score,
                                len(results) + 1,
                                fb.source,
                            )
                        )
            return results

        return list(self.fallback_retriever.search(query, top_k=top_k))


