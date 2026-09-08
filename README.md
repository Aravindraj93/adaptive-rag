# adaptive-rag

A standalone, CPU-first Python retrieval library. **0.11.0 is an experimental
release**, with an installable package and documented limitsâ€”not a claim of
production-certified adaptive retrieval.

The core is domain-agnostic and has no required third-party dependencies. It is
independent of Requirement Reader. There is no Qdrant, LangChain, LlamaIndex, GUI,
hosted service, or LLM integration.

## Install and search

From this repository, install the supplied wheel (not yet published on PyPI):

```sh
python -m pip install dist/adaptive_rag-0.11.0-py3-none-any.whl
```

```python
from adaptive_rag import BM25Retriever, Chunk

retriever = BM25Retriever()
retriever.add([
    Chunk('returns', 'policy', 'Returns are accepted within thirty days.'),
    Chunk('hours', 'guide', 'The office opens at nine in the morning.'),
])
for result in retriever.search('returns policy', top_k=3):
    print(result.chunk.id, result.score, result.chunk.text)
```

Backends accept `Chunk` objects; use `TokenChunker` for documents. File-format
parsers, embeddings, domain plugins and application authorization are supplied by
the caller. There is no implicit directory ingestion or automatic model download.

## Reliable repeated-query optimization

```python
from adaptive_rag import CachedRetriever

# Fixed token is appropriate only while this backend/configuration is immutable.
cached = CachedRetriever(
    retriever, revision=lambda: 'index-v1/config-v1', scope='my-application',
    max_entries=1024, max_bytes=16 * 1024 * 1024,
)
first = cached.search('returns policy', top_k=3)
again = cached.search('returns policy', top_k=3)  # exact result reuse
print(cached.info())
```

Cache keys include query type/text/metadata, result depth, scope and revision.
Advance a never-reused revision token whenever index contents, models, tokenizers,
configuration or access policies change. Coordinate writes with the backend; the
cache does not create a transactional index snapshot. Put tenant/permission context
in query metadata and enforce authorization outside the cache. Scope is not an ACL.
Unsupported key types bypass caching. The byte limit measures serialized payloads,
not total process RAM. A lock serializes searches; this favors correctness over
parallel miss throughput. Novel queries receive no embedding-compute reduction.

## Retrieval capabilities

| Capability | Public entry points |
| --- | --- |
| Data and chunking | `Document`, `Chunk`, `Query`, `SearchResult`, `TokenChunker` |
| Sparse retrieval | `BM25Retriever`, `MMapBM25Retriever` |
| Incremental storage | `SegmentedBM25Index`, `SegmentedBM25Retriever`, `AsyncSegmentCoordinator` |
| Dense retrieval | `DenseRetriever`, `MMapDenseRetriever`, `LSHDenseRetriever` |
| Composition | `ReciprocalRankFusionRetriever`, `SelectiveRerankingRetriever` |
| Optional confidence routing | `AdaptiveRetriever`, `EscalatingRetriever`, `AdaptiveFusionRetriever`, `FusionPolicy` |
| Safe exact reuse | `CachedRetriever`, `CacheInfo` |
| Profiling and filtering | `profile_hardware`, `TelemetryCollector`, `MetadataFilter` |

Dense retrieval requires an application-supplied embedder. `HashingEmbedder` is a
deterministic systems-test fixture, not a substitute for a semantic model.
Selective confidence routing is **opt-in**. Public evaluations did not establish
consistent quality-preserving savings for that heuristic; always-fusion remains
the conservative `FusionPolicy()` choice. Exact caching addresses repeated queries,
not the unresolved novel-query routing problem.

## Validation and supported use

See [RELEASE_REPORT.md](RELEASE_REPORT.md) for actual test counts, version coverage,
benchmarks, acceptance checks and remaining limits. See [API.md](API.md),
[COMPATIBILITY.md](COMPATIBILITY.md), and [SECURITY.md](SECURITY.md) before deployment.

The release includes a wheel and source archive. Nothing has been uploaded to a
package registry. Linux/macOS CI is supplied, but only environments explicitly
listed in the release report are verified. Index files are trusted local artifacts,
not a safe interchange format for arbitrary hostile input. Checksums detect
corruption; they do not authenticate publishers.

## Develop and reproduce

```sh
python -m pip install -e .
python -m unittest discover -s tests -q
python -m adaptive_rag.benchmark
python -m build
```

Install build tools separately when needed. Public-model evaluators require optional
numpy, onnxruntime and tokenizers packages. Their exact versions and asset hashes
are recorded in reports. [Phase 9](PHASE9.md), [Phase 10](PHASE10.md), and
[Phase 11](PHASE11.md) preserve the full calibration/evaluation history, including
negative findings. [Earlier documentation](docs/DEVELOPMENT_HISTORY.md) is historical
and may describe prior defaults. New changes are listed in [CHANGELOG.md](CHANGELOG.md).

## License

Apache-2.0; see [LICENSE](LICENSE), [NOTICE](NOTICE), and [THIRD_PARTY.md](THIRD_PARTY.md).
