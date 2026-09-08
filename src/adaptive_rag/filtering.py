"""Composable metadata filtering without domain-specific field assumptions."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping

from .models import Query, SearchResult
from .retrievers.base import Retriever


@dataclass(frozen=True, slots=True)
class MetadataFilter:
    """Match exact values, membership sets, and required keys."""

    equals: Mapping[str, Any] = field(default_factory=dict)
    any_of: Mapping[str, frozenset[Any]] = field(default_factory=dict)
    exists: frozenset[str] = field(default_factory=frozenset)

    def __call__(self, metadata: Mapping[str, Any]) -> bool:
        return self.matches(metadata)

    def matches(self, metadata: Mapping[str, Any]) -> bool:
        def encoded(value):
            return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
        if any(key not in metadata or encoded(metadata[key]) != encoded(value) for key, value in self.equals.items()):
            return False
        if any(key not in metadata or encoded(metadata[key]) not in {encoded(choice) for choice in choices} for key, choices in self.any_of.items()):
            return False
        return self.exists <= metadata.keys()


class MetadataFilteredRetriever:
    """Over-fetch from any retriever, then retain matching chunk metadata."""

    def __init__(
        self,
        retriever: Retriever,
        where: MetadataFilter,
        *,
        candidate_multiplier: int = 10,
    ) -> None:
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1")
        self.retriever = retriever
        self.where = where
        self.candidate_multiplier = candidate_multiplier

    def search(self, query: str | Query, *, top_k: int = 5) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        candidates = self.retriever.search(query, top_k=top_k * self.candidate_multiplier)
        selected = [result for result in candidates if self.where.matches(result.chunk.metadata)]
        return [
            SearchResult(result.chunk, result.score, rank, f"filtered:{result.source}")
            for rank, result in enumerate(selected[:top_k], start=1)
        ]
