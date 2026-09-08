"""Dependency-free dense retrieval shell for application-supplied embeddings."""

from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Sequence

from ..filtering import MetadataFilter
from ..models import Chunk, Query, RetrievalStats, SearchResult
from ..persistence import IndexFormatError, read_manifest, write_manifest
from ..telemetry import TelemetryCollector

Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def _unit(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    magnitude = math.sqrt(sum(value * value for value in values))
    if not values or magnitude == 0 or not math.isfinite(magnitude):
        raise ValueError('embeddings must have finite non-zero magnitude')
    return tuple(value / magnitude for value in values)


class DenseRetriever:
    """Exact cosine search using a user-supplied batch embedding function."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        min_score: float | None = 0.0,
        telemetry: TelemetryCollector | None = None,
    ) -> None:
        self.embedder = embedder
        self.min_score = min_score
        self.telemetry = telemetry or TelemetryCollector()
        self._items: list[tuple[Chunk, tuple[float, ...]]] = []
        self._dimensions: int | None = None
        self.last_stats: RetrievalStats | None = None

    def add(self, chunks: Iterable[Chunk]) -> None:
        additions = list(chunks)
        vectors = list(self.embedder([chunk.text for chunk in additions]))
        if len(vectors) != len(additions):
            raise ValueError("embedder returned a different number of vectors than texts")
        normalized = [_unit(vector) for vector in vectors]
        dimensions = {len(vector) for vector in normalized}
        if len(dimensions) > 1 or (dimensions and self._dimensions not in (None, next(iter(dimensions)))):
            raise ValueError("all embeddings must have the same dimensions")
        if dimensions:
            self._dimensions = next(iter(dimensions))
        existing = {chunk.id for chunk, _ in self._items}
        incoming = [chunk.id for chunk in additions]
        if len(incoming) != len(set(incoming)) or existing.intersection(incoming):
            raise ValueError("chunk ids must be unique")
        self._items.extend(zip(additions, normalized))

    def save(self, path: str | Path) -> None:
        """Persist normalized vectors and chunks in a checksummed JSON manifest."""

        write_manifest(
            path,
            {
                "engine": "dense-exact",
                "engine_version": 1,
                "dimensions": self._dimensions,
                "min_score": self.min_score,
                "items": [
                    {
                        "chunk": {
                            "id": chunk.id,
                            "document_id": chunk.document_id,
                            "text": chunk.text,
                            "metadata": dict(chunk.metadata),
                            "position": chunk.position,
                        },
                        "vector": list(vector),
                    }
                    for chunk, vector in self._items
                ],
            },
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        embedder: Embedder,
        telemetry: TelemetryCollector | None = None,
    ) -> "DenseRetriever":
        """Restore vectors while retaining an embedder for new chunks and queries."""

        payload = read_manifest(path)
        try:
            if payload.get("engine") != "dense-exact" or payload.get("engine_version") != 1:
                raise IndexFormatError("unsupported dense index engine")
            retriever = cls(
                embedder,
                min_score=payload.get("min_score"),
                telemetry=telemetry,
            )
            raw_items = payload["items"]
            if not isinstance(raw_items, list):
                raise IndexFormatError("invalid dense item records")
            ids: set[str] = set()
            for item in raw_items:
                chunk_value = item["chunk"]
                chunk = Chunk(
                    chunk_value["id"],
                    chunk_value["document_id"],
                    chunk_value["text"],
                    chunk_value.get("metadata", {}),
                    int(chunk_value.get("position", 0)),
                )
                if chunk.id in ids:
                    raise IndexFormatError("duplicate chunk id in dense index")
                ids.add(chunk.id)
                vector = _unit(item["vector"])
                if retriever._dimensions is None:
                    retriever._dimensions = len(vector)
                elif len(vector) != retriever._dimensions:
                    raise IndexFormatError("inconsistent dense vector dimensions")
                retriever._items.append((chunk, vector))
            expected_dimensions = payload.get("dimensions")
            if expected_dimensions is not None and int(expected_dimensions) != retriever._dimensions:
                raise IndexFormatError("dense dimension metadata mismatch")
            return retriever
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, IndexFormatError):
                raise
            raise IndexFormatError(f"invalid dense index payload: {error}") from error

    def search(
        self,
        query: str | Query,
        *,
        top_k: int = 5,
        where: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        started = perf_counter()
        text = query.text if isinstance(query, Query) else query
        vectors = list(self.embedder([text]))
        if len(vectors) != 1:
            raise ValueError("embedder must return exactly one query vector")
        query_vector = _unit(vectors[0])
        if self._dimensions is not None and len(query_vector) != self._dimensions:
            raise ValueError("query embedding dimensions do not match the index")
        eligible = self._items
        if where is not None:
            eligible = [item for item in eligible if where.matches(item[0].metadata)]
        scored = [
            (sum(left * right for left, right in zip(query_vector, vector)), chunk)
            for chunk, vector in eligible
        ]
        if self.min_score is not None:
            scored = [item for item in scored if item[0] > self.min_score]
        scored.sort(key=lambda item: (-item[0], item[1].id))
        results = [
            SearchResult(chunk, score, rank, "dense")
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]
        latency_ms = (perf_counter() - started) * 1000
        self.last_stats = RetrievalStats(1, len(self._items), len(results), latency_ms, "dense")
        self.telemetry.record("dense.search", latency_ms, candidates=len(self._items), returned=len(results))
        return results
