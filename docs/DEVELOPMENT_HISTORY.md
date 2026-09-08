# adaptive-rag

Phase 11 adds stricter, uncertainty-checked calibration and fresh ArguAna transfer
evaluation. The guard rejects the selective proposal; no runtime defaults change.
See [Phase 11](PHASE11.md) for the evidence and limitations.

Version 0.10.0rc1 adds opt-in candidate-reusing adaptive fusion and explicit,
development-calibrated feature weights. Existing router defaults remain unchanged.
See [Phase 10](PHASE10.md) for the new API, fresh-data evaluation and limitations.

Phase 9 adds held-out routing calibration and live-embedding CPU evaluation on
Cranfield and SciFact, without changing the library API or defaults.
See [Phase 9 results and protocol](PHASE9.md).

Version 0.8.0rc1 is a release candidate with reader-aware segment cleanup,
filter parity fixes, and public-corpus evaluation. See [Phase 8](PHASE8.md) for
validation results, reproducible evaluation instructions, and known limits.

Version 0.7 adds process-level segment writer locks, journaled manifest publication,
reader-safe compaction, and checksummed LSH persistence. See [Phase 7 notes](PHASE7.md)
for recovery behavior and storage limits. The sections below describe the underlying
retrieval backends and retain the measured Phase 6 benchmark results.

`adaptive-rag` is a CPU-first, domain-agnostic Python retrieval library. Version 0.6 originally
adds indexed metadata facets, serialized background index operations, and approximate
cosine retrieval with random-hyperplane LSH. It has no required third-party runtime
dependencies.

The core contains no automotive, legal, healthcare, or other application concepts.
Qdrant, LangChain, LlamaIndex, GUIs, and LLM calls remain outside the project.

## Capabilities

- Immutable document, chunk, query, result, and telemetry contracts
- Deterministic overlapping chunking with stable ids and offsets
- In-memory and memory-mapped Okapi BM25
- Disk-backed SQLite staging for sparse index construction
- Persistent indexed metadata facets applied before sparse scoring
- Segment-based append and tombstone deletion with explicit compaction
- Ordered background append, delete, and compaction jobs
- Exact dense search with JSON or binary float32 memory-mapped vectors
- Approximate dense candidate search using deterministic multi-table LSH
- Dependency-free `HashingEmbedder` for systems and fusion baselines
- Native metadata filtering across sparse, exact-dense, and LSH backends
- Hardware budgets, confidence routing, RRF, selective reranking, and calibration
- Domain-neutral plugin protocols and reproducible benchmarks

## Install

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Python 3.10+ is supported.

## Sparse retrieval with facet pushdown

```python
from adaptive_rag import MMapBM25Retriever, MetadataFilter

MMapBM25Retriever.build("search-index", chunk_iterator).close()

with MMapBM25Retriever("search-index") as retriever:
    results = retriever.search(
        "return policy",
        top_k=5,
        where=MetadataFilter(
            equals={"language": "en"},
            any_of={"year": frozenset({2025, 2026})},
            exists=frozenset({"source"}),
        ),
    )
```

During construction, metadata values are encoded canonically in an indexed SQLite
facet database. Equality, membership, and existence filters resolve eligible chunk
ordinals before BM25 scoring. The facet database is read-only while the index is open
and is covered by the index checksum manifest.

Sparse posting triples use a separate temporary SQLite stage and external ordering,
so the builder does not retain the complete posting dictionary in Python memory.

## Incremental segments and background operations

```python
from adaptive_rag import (
    AsyncSegmentCoordinator,
    Chunk,
    SegmentedBM25Index,
    SegmentedBM25Retriever,
)

SegmentedBM25Index.create("segments", initial_chunks)

with AsyncSegmentCoordinator("segments") as coordinator:
    append_job = coordinator.append([Chunk("new", "source", "new content")])
    append_job.result()
    coordinator.delete(["obsolete"]).result()
    coordinator.compact().result()

with SegmentedBM25Retriever("segments") as retriever:
    results = retriever.search("new content")
```

Each append creates an immutable segment. Deletes are durable tombstones. Search
combines segment-local rankings with reciprocal-rank fusion; compaction rebuilds one
segment containing only live chunks.

`AsyncSegmentCoordinator` uses one worker thread, so mutations submitted through one
coordinator execute in order without blocking the caller. It does not provide locking
between separate processes or separate coordinator instances.

## Exact and approximate dense retrieval

```python
from adaptive_rag import HashingEmbedder, LSHDenseRetriever, MMapDenseRetriever

embedder = HashingEmbedder(dimensions=256)

MMapDenseRetriever.build(
    "dense-index",
    chunks,
    embedder=embedder,
    batch_size=64,
).close()

approximate = LSHDenseRetriever(
    embedder,
    dimensions=256,
    tables=8,
    bits=10,
    probe_radius=1,
    seed=17,
)
approximate.add(chunks)
```

`MMapDenseRetriever` performs exact cosine search over checksummed float32 vectors.
`LSHDenseRetriever` uses random-hyperplane signatures and optionally probes buckets at
Hamming distance one, then performs exact cosine scoring only within the candidate
set. LSH trades recall for fewer scored candidates; exact dense search remains the
reference implementation.

`HashingEmbedder` captures normalized token overlap. It is deterministic and useful
for integration tests, but it is not a trained semantic model. Any batch embedding
callable can be supplied instead.

## Composition

- `MetadataFilter` supports exact values, membership sets, and required keys.
- `EscalatingRetriever` falls back when the primary confidence gate rejects a result.
- `ReciprocalRankFusionRetriever` combines rankings with incomparable score scales.
- `SelectiveRerankingRetriever` reranks only ambiguous leading candidates.
- `calibrate_confidence` selects a threshold against labelled cases.
- `PluginPipeline` applies generic enrichers and query analyzers without silent
  metadata-key replacement.

## Benchmarks

Sparse storage benchmark:

```powershell
$env:PYTHONPATH = "src"
python -m adaptive_rag.scale_benchmark `
  --documents 10000 `
  --queries 200 `
  --output benchmark-phase6-sparse.json
```

| Backend | Retained Python memory | Peak build memory | Build | Mean query | Top-1 |
|---|---:|---:|---:|---:|---:|
| In-memory BM25 | 22.75 MB | 23.36 MB | 778 ms | 0.0164 ms | 100% |
| Memory-mapped BM25 + facets | 8.24 MB | 23.57 MB | 7,836 ms | 0.0217 ms | 100% |

The facet database increased the sample index from 4.42 MB to 5.02 MB and startup to
about 64.8 ms. Runtime traced Python memory remained approximately 64% below the
in-memory backend.

Approximate dense benchmark:

```powershell
python -m adaptive_rag.ann_benchmark `
  --documents 5000 `
  --queries 100 `
  --dimensions 256 `
  --output benchmark-phase6-ann.json
```

| Mode | Mean query | Quality | Mean candidates |
|---|---:|---:|---:|
| Exact dense | 74.6 ms | 98% synthetic top-1 | 5,000 |
| LSH dense | 23.5 ms | 96% recall@1 vs exact | 1,371 (27.4%) |

Both datasets are synthetic and lexically simple. These results characterize storage
and candidate-selection behavior, not production semantic quality.

## Tests

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## Architecture

```text
Documents -> plugins -> chunking
                       |-> BM25 -> postings mmap + indexed facets
                       |-> immutable BM25 segments -> tombstones -> compaction
                       |                              ^
                       |                         ordered background worker
                       |-> embedder -> exact float32 mmap vectors
                                    -> LSH buckets -> candidate cosine scoring

Query -> native filters -> confidence routing -> RRF -> optional reranker
                                             -> telemetry and calibration
```

## Next phase

Recommended Phase 7 work:

1. Replace large JSON chunk/term offset arrays with compact binary tables.
2. Add inter-process writer locks and crash-recovery journals for segment mutations.
3. Add configurable facet allowlists to control high-cardinality index growth.
4. Provide an optional maintained ONNX/static embedding adapter.
5. Persist and incrementally update LSH tables.
6. Evaluate lexical, dense, fused, and routed modes on harder public datasets.

## License

Apache-2.0.
