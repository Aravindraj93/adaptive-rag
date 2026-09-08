"""Bounded exact-result reuse for explicitly revisioned, deterministic retrieval."""
from __future__ import annotations

import copy
import json
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Callable

from .models import Chunk, Query, SearchResult
from .retrievers.base import Retriever


def _json(value: object) -> bytes:
    def validate(item):
        if type(item) in (str, int, float, bool, type(None)):
            return
        if type(item) is list:
            for child in item:
                validate(child)
            return
        if type(item) is dict and all(type(k) is str for k in item):
            for child in item.values():
                validate(child)
            return
        raise TypeError('only strict JSON values can be cached')
    validate(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(',', ':')).encode('utf-8')


def _clone(rows):
    return [SearchResult(Chunk(r.chunk.id, r.chunk.document_id, r.chunk.text,
                              copy.deepcopy(dict(r.chunk.metadata)), r.chunk.position),
                         r.score, r.rank, r.source) for r in rows]


@dataclass(frozen=True, slots=True)
class CacheInfo:
    hits: int
    misses: int
    bypasses: int
    evictions: int
    entries: int
    serialized_bytes: int


class CachedRetriever:
    """Exact LRU cache, never semantic/fuzzy matching.

    `revision` must be a monotonic string/int token covering corpus, model,
    tokenizer, filters, access policy and all retrieval configuration. Callers
    must coordinate mutations with the backend and advance the revision before
    publishing new state. Never recycle revision tokens (ABA is not detected).
    `scope` identifies the security/application namespace. Include per-query
    tenant/permission context in Query.metadata; a scope is NOT authorization.

    A single reentrant lock serializes searches, avoiding duplicate concurrent
    misses and allowing non-thread-safe backends. Changing revision during a
    search prevents caching but cannot make a mutating backend transactional.
    Byte limits cover serialized payloads, not total Python heap. Cached results
    are deep copied on ingress and egress. Unsupported query keys bypass reuse.
    `clear()` is a synchronization barrier but does not change backend revision.
    This wrapper intentionally exposes only the common query/top_k interface.
    """
    def __init__(self, backend: Retriever, *, revision: Callable[[], str | int],
                 scope: str, max_entries: int = 1024, max_bytes: int = 16*1024*1024):
        if not isinstance(scope, str) or not scope:
            raise ValueError('a nonempty application/security scope is required')
        if (type(max_entries) is not int or max_entries < 1
                or type(max_bytes) is not int or max_bytes < 1):
            raise ValueError('positive integer cache limits required')
        if not callable(revision):
            raise TypeError('revision must be callable')
        self.backend, self.revision, self.scope = backend, revision, scope
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._lock = RLock()
        self._items = OrderedDict()
        self._generation = None
        self._bytes = self._hits = self._misses = self._bypasses = self._evictions = 0

    def _revision(self):
        token = self.revision()
        if type(token) not in (str, int):
            raise TypeError('revision must return a string or integer')
        return (type(token).__name__, token)

    def clear(self):
        with self._lock:
            self._items.clear()
            self._bytes = 0

    def info(self) -> CacheInfo:
        with self._lock:
            return CacheInfo(self._hits, self._misses, self._bypasses, self._evictions,
                             len(self._items), self._bytes)

    def search(self, query: str | Query, *, top_k: int = 5) -> list[SearchResult]:
        if type(top_k) is not int or top_k < 1:
            raise ValueError('top_k must be a positive integer')
        with self._lock:
            generation = self._revision()
            if isinstance(query, Query):
                try:
                    query = Query(query.text, copy.deepcopy(dict(query.metadata)))
                except (TypeError, ValueError, RecursionError):
                    self._bypasses += 1
                    return list(self.backend.search(query, top_k=top_k))
            if generation != self._generation:
                self.clear()
                self._generation = generation
            try:
                # Keep str and Query distinct; preserve missing/null and JSON types.
                key = _json([self.scope, list(generation), top_k,
                             ['query', query.text, dict(query.metadata)] if isinstance(query, Query)
                             else ['string', query]])
            except (TypeError, ValueError, OverflowError, RecursionError):
                self._bypasses += 1
                return list(self.backend.search(query, top_k=top_k))
            if key in self._items:
                rows, size = self._items.pop(key)
                self._items[key] = (rows, size)
                # Recheck external revision before serving a hit.
                if self._revision() == generation:
                    self._hits += 1
                    return _clone(rows)
                self.clear()
                generation = self._revision()
                self._generation = generation
                # The old key cannot be used with the new generation.
                self._bypasses += 1
                return list(self.backend.search(query, top_k=top_k))
            self._misses += 1
            rows = list(self.backend.search(query, top_k=top_k))
            if self._revision() != generation:
                self.clear()
                self._bypasses += 1
                return rows
            try:
                saved = _clone(rows)
                size = len(key)+len(_json([dict(id=r.chunk.id, document_id=r.chunk.document_id,
                    text=r.chunk.text, metadata=dict(r.chunk.metadata), position=r.chunk.position,
                    score=r.score, rank=r.rank, source=r.source) for r in saved]))
            except (TypeError, ValueError, OverflowError, RecursionError):
                self._bypasses += 1
                return rows
            if size > self.max_bytes:
                self._bypasses += 1
                return rows
            while self._items and (len(self._items) >= self.max_entries or self._bytes+size > self.max_bytes):
                _, (_, removed) = self._items.popitem(last=False)
                self._bytes -= removed
                self._evictions += 1
            self._items[key] = (saved, size)
            self._bytes += size
            return _clone(saved)
