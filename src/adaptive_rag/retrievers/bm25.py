"""A compact Okapi BM25 baseline optimized for correctness and portability."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

from ..chunking import TokenChunker
from ..filtering import MetadataFilter
from ..models import Chunk, Document, Query, RetrievalStats, SearchResult
from ..persistence import IndexFormatError, read_manifest, write_manifest
from ..telemetry import TelemetryCollector
from ..tokenization import NormalizedTokenizer

Tokenizer = Callable[[str], list[str]]
_DEFAULT_TOKENIZER = NormalizedTokenizer()


def default_tokenizer(text: str) -> list[str]:
    """Unicode tokenizer with conservative dependency-free suffix normalization."""

    return _DEFAULT_TOKENIZER(text)


class BM25Retriever:
    """In-memory Okapi BM25 retriever with optional durable persistence."""

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Tokenizer = default_tokenizer,
        telemetry: TelemetryCollector | None = None,
    ) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be positive and b must be between 0 and 1")
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.telemetry = telemetry or TelemetryCollector()
        self._chunks: dict[str, Chunk] = {}
        self._term_frequencies: dict[str, Counter[str]] = {}
        self._document_frequencies: Counter[str] = Counter()
        self._postings: dict[str, set[str]] = defaultdict(set)
        self._total_terms = 0
        self.last_stats: RetrievalStats | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def average_document_length(self) -> float:
        return self._total_terms / len(self) if self._chunks else 0.0

    def add(self, chunks: Iterable[Chunk]) -> None:
        """Add chunks to the index. Chunk ids must be unique."""

        additions = list(chunks)
        seen: set[str] = set()
        for chunk in additions:
            if chunk.id in self._chunks or chunk.id in seen:
                raise ValueError(f"duplicate chunk id: {chunk.id}")
            seen.add(chunk.id)

        started = perf_counter()
        for chunk in additions:
            frequencies = Counter(self.tokenizer(chunk.text))
            self._chunks[chunk.id] = chunk
            self._term_frequencies[chunk.id] = frequencies
            self._total_terms += sum(frequencies.values())
            for term in frequencies:
                self._document_frequencies[term] += 1
                self._postings[term].add(chunk.id)
        self.telemetry.record(
            "bm25.index",
            (perf_counter() - started) * 1000,
            chunks=len(additions),
            index_size=len(self),
        )

    def add_documents(
        self,
        documents: Iterable[Document],
        *,
        chunker: TokenChunker | None = None,
    ) -> None:
        """Chunk and index documents with a deterministic policy."""

        self.add((chunker or TokenChunker()).chunk_many(documents))

    def save(self, path: str | Path) -> None:
        """Atomically persist chunks and settings in a checksummed manifest."""

        if self.tokenizer is default_tokenizer:
            tokenizer_config = _DEFAULT_TOKENIZER.to_dict()
        elif isinstance(self.tokenizer, NormalizedTokenizer):
            tokenizer_config = self.tokenizer.to_dict()
        else:
            raise TypeError("custom tokenizers cannot be persisted; use NormalizedTokenizer")
        chunks = [
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "metadata": dict(chunk.metadata),
                "position": chunk.position,
            }
            for chunk in sorted(self._chunks.values(), key=lambda item: item.id)
        ]
        write_manifest(
            path,
            {
                "engine": "bm25",
                "engine_version": 1,
                "configuration": {"k1": self.k1, "b": self.b, "tokenizer": tokenizer_config},
                "chunks": chunks,
            },
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        telemetry: TelemetryCollector | None = None,
    ) -> "BM25Retriever":
        """Load an index after validating its schema, engine, and checksum."""

        payload = read_manifest(path)
        try:
            if payload.get("engine") != "bm25" or payload.get("engine_version") != 1:
                raise IndexFormatError("unsupported retrieval engine")
            configuration = payload["configuration"]
            raw_chunks = payload["chunks"]
            if not isinstance(configuration, dict) or not isinstance(raw_chunks, list):
                raise IndexFormatError("invalid BM25 index payload")
            tokenizer_value = configuration["tokenizer"]
            if not isinstance(tokenizer_value, dict):
                raise IndexFormatError("invalid tokenizer configuration")
            retriever = cls(
                k1=float(configuration["k1"]),
                b=float(configuration["b"]),
                tokenizer=NormalizedTokenizer.from_dict(tokenizer_value),
                telemetry=telemetry,
            )
            chunks = []
            for item in raw_chunks:
                if not isinstance(item, dict):
                    raise IndexFormatError("invalid chunk entry")
                chunks.append(
                    Chunk(
                        id=item["id"],
                        document_id=item["document_id"],
                        text=item["text"],
                        metadata=item.get("metadata", {}),
                        position=int(item.get("position", 0)),
                    )
                )
            retriever.add(chunks)
            return retriever
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, IndexFormatError):
                raise
            raise IndexFormatError(f"invalid BM25 index payload: {error}") from error

    def search(
        self,
        query: str | Query,
        *,
        top_k: int = 5,
        where: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """Return the highest-scoring chunks, excluding zero-score matches."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        started = perf_counter()
        query_text = query.text if isinstance(query, Query) else query
        terms = self.tokenizer(query_text)
        candidates: set[str] = set()
        for term in terms:
            candidates.update(self._postings.get(term, ()))

        if where is not None:
            candidates = {
                chunk_id
                for chunk_id in candidates
                if where.matches(self._chunks[chunk_id].metadata)
            }
        scored = [(self._score(chunk_id, terms), chunk_id) for chunk_id in candidates]
        scored = [(score, chunk_id) for score, chunk_id in scored if score > 0]
        scored.sort(key=lambda item: (-item[0], item[1]))
        results = [
            SearchResult(self._chunks[chunk_id], score, rank, "bm25")
            for rank, (score, chunk_id) in enumerate(scored[:top_k], start=1)
        ]
        latency_ms = (perf_counter() - started) * 1000
        self.last_stats = RetrievalStats(
            len(terms), len(candidates), len(results), latency_ms, "bm25"
        )
        self.telemetry.record(
            "bm25.search",
            latency_ms,
            query_terms=len(terms),
            candidates=len(candidates),
            returned=len(results),
        )
        return results

    def _score(self, chunk_id: str, terms: list[str]) -> float:
        frequencies = self._term_frequencies[chunk_id]
        document_length = sum(frequencies.values())
        average_length = self.average_document_length or 1.0
        score = 0.0
        for term in set(terms):
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            document_frequency = self._document_frequencies[term]
            inverse_document_frequency = math.log(
                1 + (len(self) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            normalization = frequency + self.k1 * (
                1 - self.b + self.b * document_length / average_length
            )
            score += inverse_document_frequency * frequency * (self.k1 + 1) / normalization
        return score