"""Domain-neutral data contracts shared by retrieval implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


def _metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True, slots=True)
class Document:
    """A source document supplied by an application."""

    id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("document id must not be empty")
        if not isinstance(self.text, str):
            raise TypeError("document text must be a string")
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable passage derived from a document."""

    id: str
    document_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    position: int = 0

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.document_id.strip():
            raise ValueError("chunk id and document_id must not be empty")
        if self.position < 0:
            raise ValueError("chunk position must be non-negative")
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class Query:
    """A retrieval request, with optional application-provided context."""

    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("query text must not be empty")
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One ranked retrieval result."""

    chunk: Chunk
    score: float
    rank: int
    source: str

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank is one-based")


@dataclass(frozen=True, slots=True)
class RetrievalStats:
    """Summary telemetry for a single search."""

    query_terms: int
    candidates: int
    returned: int
    latency_ms: float
    strategy: str

