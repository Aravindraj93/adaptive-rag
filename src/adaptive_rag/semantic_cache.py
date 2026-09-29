"""Opt-in approximate retrieval reuse with explicit context and revision contracts."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import copy
import math
from threading import RLock
from time import monotonic

from .cache import _clone, _json
from .models import Query


@dataclass(frozen=True)
class SemanticCacheInfo:
    exact_hits: int
    semantic_hits: int
    misses: int
    bypasses: int
    shadow_matches: int
    shadow_disagreements: int
    entries: int
    serialized_bytes: int


class SemanticCachedRetriever:
    """Bounded result cache; exact mode is the default.

    revision() must cover corpus, embedder, permissions and retrieval settings.
    Never recycle revision tokens. Put caller-specific authorization context in
    Query.metadata or backend kwargs; scope alone does not enforce access.
    Mutations must be coordinated externally with searches and revision changes.

    shadow mode computes proposed approximate hits but ALWAYS serves the backend
    on an exact miss. Disagreement measures ranked IDs, not relevance or truth.
    semantic mode requires an application reuse_validator(old_text, new_text).
    Neither that callback nor a cosine threshold guarantees equivalent meaning.
    Use exact mode for sensitive workloads. Embeddings must be deterministic.

    Calls are serialized, results copied, TTL uses a monotonic clock. Limits
    cover serialized entries, not total process RAM. The cache is process-local.
    This replaces the unreleased ZIP API; it does not accept its SemanticCache.
    """

    def __init__(self, backend, *, revision, scope, mode="exact", embedder=None,
                 threshold=0.95, reuse_validator=None, max_entries=1024,
                 max_bytes=16 * 1024 * 1024, ttl_seconds=300.0):
        if mode not in ("exact", "shadow", "semantic"):
            raise ValueError("mode must be exact, shadow or semantic")
        if not callable(revision) or not isinstance(scope, str) or not scope:
            raise ValueError("revision callback and nonempty scope required")
        if mode != "exact" and not callable(embedder):
            raise ValueError("shadow/semantic mode requires a query embedding callable")
        if mode == "semantic" and not callable(reuse_validator):
            raise ValueError("semantic reuse requires an application reuse_validator")
        if reuse_validator is not None and not callable(reuse_validator):
            raise TypeError("reuse_validator must be callable")
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be finite and between zero and one")
        if any(type(v) is not int or v < 1 for v in (max_entries, max_bytes)):
            raise ValueError("positive integer cache limits required")
        if type(ttl_seconds) not in (int, float) or not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive and finite")
        self.backend, self.revision, self.scope = backend, revision, scope
        self.mode, self.embedder, self.threshold = mode, embedder, threshold
        self.reuse_validator = reuse_validator
        self.max_entries, self.max_bytes, self.ttl_seconds = max_entries, max_bytes, ttl_seconds
        self._lock, self._items = RLock(), OrderedDict()
        self._generation = None
        self._bytes = 0
        self._counts = dict(exact_hits=0, semantic_hits=0, misses=0, bypasses=0,
                            shadow_matches=0, shadow_disagreements=0)

    def _revision(self):
        value = self.revision()
        if type(value) not in (str, int):
            raise TypeError("revision must return a string or integer")
        return (type(value).__name__, value)

    def clear(self):
        """Drop entries without resetting cumulative counters."""
        with self._lock:
            self._items.clear()
            self._bytes = 0

    def info(self):
        with self._lock:
            self._expire()
            return SemanticCacheInfo(**self._counts, entries=len(self._items),
                                     serialized_bytes=self._bytes)

    def _expire(self):
        now = monotonic()
        for key, entry in list(self._items.items()):
            if now >= entry[4]:
                self._bytes -= self._items.pop(key)[5]

    def _vector(self, text):
        values = tuple(float(v) for v in self.embedder(text))
        norm = math.hypot(*values)
        if not values or not all(math.isfinite(v) for v in values) or not math.isfinite(norm) or norm == 0:
            raise ValueError("embedder must return a finite nonzero vector")
        return tuple(v / norm for v in values)

    def search(self, query, *, top_k=5, **kwargs):
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(query, (str, Query)):
            raise TypeError("query must be str or Query")
        with self._lock:
            generation = self._revision()
            if generation != self._generation:
                self.clear()
                self._generation = generation
            self._expire()
            text = query.text if isinstance(query, Query) else query
            try:
                metadata = dict(query.metadata) if isinstance(query, Query) else None
                # Validate before copying arbitrary caller-provided objects.
                context = _json([self.scope, list(generation), top_k, metadata, kwargs])
                kwargs = copy.deepcopy(kwargs)
                if isinstance(query, Query):
                    query = Query(text, copy.deepcopy(metadata))
                key = (context, text)
            except (TypeError, ValueError, OverflowError, RecursionError):
                self._counts["bypasses"] += 1
                return list(self.backend.search(query, top_k=top_k, **kwargs))
            entry = self._items.get(key)
            if entry is not None and self._revision() == generation:
                self._items.move_to_end(key)
                self._counts["exact_hits"] += 1
                return _clone(entry[3])
            vector, candidate = None, None
            if self.mode != "exact":
                vector = self._vector(text)
                best = -1.0
                for (other_context, other_text), stored in self._items.items():
                    if other_context != context or len(stored[2]) != len(vector):
                        continue
                    similarity = min(1.0, sum(a*b for a, b in zip(vector, stored[2])))
                    if similarity >= self.threshold and similarity > best:
                        if self.reuse_validator is not None and self.reuse_validator(other_text, text) is not True:
                            continue
                        best, candidate = similarity, stored
            if (candidate is not None and self.mode == "semantic"
                    and monotonic() < candidate[4] and self._revision() == generation):
                self._counts["semantic_hits"] += 1
                return _clone(candidate[3])
            self._counts["misses"] += 1
            rows = list(self.backend.search(query, top_k=top_k, **kwargs))
            if self._revision() != generation:
                self.clear()
                self._counts["bypasses"] += 1
                return rows
            if candidate is not None and self.mode == "shadow":
                self._counts["shadow_matches"] += 1
                if [r.chunk.id for r in candidate[3]] != [r.chunk.id for r in rows]:
                    self._counts["shadow_disagreements"] += 1
            try:
                saved = _clone(rows)
                payload = [dict(id=r.chunk.id, document_id=r.chunk.document_id,
                                text=r.chunk.text, metadata=dict(r.chunk.metadata),
                                position=r.chunk.position, score=r.score,
                                rank=r.rank, source=r.source) for r in saved]
                size = len(context) + len(_json([text, list(vector or ()), payload]))
            except (TypeError, ValueError, OverflowError, RecursionError):
                self._counts["bypasses"] += 1
                return rows
            if size > self.max_bytes:
                self._counts["bypasses"] += 1
                return rows
            while self._items and (len(self._items) >= self.max_entries or self._bytes + size > self.max_bytes):
                self._bytes -= self._items.popitem(last=False)[1][5]
            if key in self._items:
                self._bytes -= self._items.pop(key)[5]
            self._items[key] = (context, text, vector, saved, monotonic() + self.ttl_seconds, size)
            self._bytes += size
            return _clone(saved)
