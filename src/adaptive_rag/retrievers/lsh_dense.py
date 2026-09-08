"""Pure-Python random-hyperplane LSH for approximate cosine candidates."""

from __future__ import annotations

import random
import math
from pathlib import Path
from collections import defaultdict
from time import perf_counter
from typing import Iterable

from ..filtering import MetadataFilter
from ..models import Chunk, Query, RetrievalStats, SearchResult
from ..telemetry import TelemetryCollector
from ..persistence import read_manifest, write_manifest, IndexFormatError
from .dense import Embedder, _unit


class LSHDenseRetriever:
    """Approximate dense retrieval with deterministic multi-table sign hashes."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        dimensions: int,
        tables: int = 6,
        bits: int = 10,
        probe_radius: int = 1,
        seed: int = 0,
        min_score: float | None = 0.0,
        telemetry: TelemetryCollector | None = None,
    ) -> None:
        if dimensions < 1 or tables < 1 or bits < 1:
            raise ValueError("dimensions, tables, and bits must be positive")
        if probe_radius not in (0, 1):
            raise ValueError("probe_radius currently supports only 0 or 1")
        self.embedder = embedder
        self.dimensions = dimensions
        self.tables = tables
        self.bits = bits
        self.probe_radius = probe_radius
        self.min_score = min_score
        self.telemetry = telemetry or TelemetryCollector()
        generator = random.Random(seed)
        self._planes = [
            [tuple(generator.gauss(0, 1) for _ in range(dimensions)) for _ in range(bits)]
            for _ in range(tables)
        ]
        self._buckets = [defaultdict(list) for _ in range(tables)]
        self._items: list[tuple[Chunk, tuple[float, ...]]] = []
        self.last_stats: RetrievalStats | None = None

    def add(self, chunks: Iterable[Chunk]) -> None:
        additions = list(chunks)
        ids = {chunk.id for chunk, _ in self._items}
        incoming = [chunk.id for chunk in additions]
        if len(incoming) != len(set(incoming)) or ids.intersection(incoming):
            raise ValueError("chunk ids must be unique")
        vectors = list(self.embedder([chunk.text for chunk in additions]))
        if len(vectors) != len(additions):
            raise ValueError("embedder returned a different number of vectors than texts")
        normalized = [_unit(vector) for vector in vectors]
        if any(len(vector) != self.dimensions or not all(map(math.isfinite, vector)) for vector in normalized):
            raise ValueError('invalid embedding dimensions or nonfinite values')
        for chunk, vector in zip(additions, normalized):
            ordinal = len(self._items)
            self._items.append((chunk, vector))
            for table in range(self.tables):
                self._buckets[table][self._signature(vector, table)].append(ordinal)

    def save(self, path):
        write_manifest(path, {
            'engine': 'lsh-dense', 'engine_version': 1,
            'configuration': dict(dimensions=self.dimensions, tables=self.tables,
                                  bits=self.bits, probe_radius=self.probe_radius, min_score=self.min_score),
            'planes': self._planes,
            'items': [dict(chunk=dict(id=c.id, document_id=c.document_id, text=c.text,
                                     metadata=dict(c.metadata), position=c.position), vector=v)
                      for c, v in self._items],
        })

    @classmethod
    def load(cls, path, *, embedder):
        """Restore stored vectors and rebuild buckets without calling the embedder."""
        payload = read_manifest(path)
        try:
            if payload['engine'] != 'lsh-dense' or payload['engine_version'] != 1:
                raise ValueError('unsupported LSH format')
            index = cls(embedder, **payload['configuration'])
            planes = payload['planes']
            if len(planes) != index.tables or any(len(t) != index.bits for t in planes):
                raise ValueError('invalid plane counts')
            if any(len(p) != index.dimensions or not all(math.isfinite(x) for x in p)
                   for t in planes for p in t):
                raise ValueError('invalid planes')
            index._planes = planes
            seen = set()
            for item in payload['items']:
                chunk = Chunk(**item['chunk'])
                vector = tuple(item['vector'])
                if chunk.id in seen or len(vector) != index.dimensions:
                    raise ValueError('duplicate id or invalid vector dimensions')
                if not all(map(math.isfinite, vector)) or not math.isclose(sum(v*v for v in vector), 1.0, abs_tol=1e-6):
                    raise ValueError('invalid normalized vector')
                seen.add(chunk.id)
                ordinal = len(index._items)
                index._items.append((chunk, vector))
                for table in range(index.tables):
                    index._buckets[table][index._signature(vector, table)].append(ordinal)
            return index
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise IndexFormatError(f'invalid LSH index: {error}') from error

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
        embedded = list(self.embedder([text]))
        if len(embedded) != 1:
            raise ValueError("embedder must return exactly one query vector")
        vector = _unit(embedded[0])
        if len(vector) != self.dimensions:
            raise ValueError("query embedding dimensions do not match the LSH index")
        candidates: set[int] = set()
        for table in range(self.tables):
            signature = self._signature(vector, table)
            candidates.update(self._buckets[table].get(signature, ()))
            if self.probe_radius == 1:
                for bit in range(self.bits):
                    candidates.update(self._buckets[table].get(signature ^ (1 << bit), ()))
        scored = []
        for ordinal in candidates:
            chunk, candidate_vector = self._items[ordinal]
            if where is not None and not where.matches(chunk.metadata):
                continue
            score = sum(left * right for left, right in zip(vector, candidate_vector))
            if self.min_score is None or score > self.min_score:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        results = [
            SearchResult(chunk, score, rank, "lsh-dense")
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]
        latency_ms = (perf_counter() - started) * 1000
        self.last_stats = RetrievalStats(1, len(candidates), len(results), latency_ms, "lsh-dense")
        self.telemetry.record(
            "lsh-dense.search", latency_ms, candidates=len(candidates), returned=len(results)
        )
        return results

    def _signature(self, vector: tuple[float, ...], table: int) -> int:
        signature = 0
        for bit, plane in enumerate(self._planes[table]):
            if sum(value * normal for value, normal in zip(vector, plane)) >= 0:
                signature |= 1 << bit
        return signature
