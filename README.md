# adaptive-rag

CPU-first retrieval primitives and evaluation for existing RAG pipelines.
The core is domain-agnostic, with no mandatory third-party dependencies.

**Current development candidate: 0.17.0rc1.** This is not a production-readiness
certification or a claim of universal speedups. It builds on upstream 0.12.0;
unreleased experimental ZIP features are not included wholesale.

## Install from this checkout

```sh
python -m pip install .
```

No PyPI publication of this candidate is implied. Optional extras are `pdf`,
`images`, and `numpy`. OCR also needs a separately installed Tesseract executable.
Bring your own embedding model for dense/hybrid retrieval; model dependencies,
download sizes and licenses are separate from this library.

## Working CPU-only example

```python
from adaptive_rag import BM25Retriever, Chunk, SemanticCachedRetriever

backend = BM25Retriever()
backend.add([Chunk("refund-policy", "handbook", "Refunds are available within thirty days.")])
revision = 1
retriever = SemanticCachedRetriever(
    backend, scope="public-handbook", revision=lambda: revision,
)  # exact caching by default; no embedding required
print(retriever.search("refund period", top_k=3))
print(retriever.search("refund period", top_k=3))
print(retriever.info())
```

Advance the revision whenever corpus, permissions, model, or retrieval settings
change. Coordinate updates with readers. Include caller-specific permissions and
filters in Query.metadata or supported backend arguments. A cache is not an
authorization system.

## Compare before and after

```sh
adaptive-rag-evaluate --repetitions 5 --output comparison.json
adaptive-rag-evaluate --dataset benchmarks/general_knowledge.json --repetitions 5
```

The CLI compares BM25 with and without exact caching on identical inputs.
The built-in six-query fixture is a smoke test, not a representative benchmark.
Reports include recall/nDCG for labelled queries, median/p95 latency, first versus
repeated passes, cache counts and individual ranking regressions. They do not
measure RAM, indexing cost, embedding calls or dollar savings. The Python
`compare_retrievers` function accepts your existing baseline and candidate.

Without labels, quality metrics are null: agreement does not prove relevance.
Repeated-query benefits depend on your actual workload; caching can be slower
than a cheap underlying search. Reports omit query text but include chunk IDs;
review them for sensitive information before sharing.

## Semantic cache: evaluate before enabling

Exact mode is the default. `mode="shadow"` requires a deterministic query
embedding callable and measures proposed semantic hits without serving them.
`mode="semantic"` additionally requires an application `reuse_validator`.
Thresholds and validators do not guarantee semantic equivalence. Negation,
numbers, dates and permissions need explicit tests. For sensitive applications,
keep exact mode. See [focused release guide](docs/FOCUSED_RELEASE.md).

## Available building blocks

- In-memory and memory-mapped BM25, dense retrieval and hybrid rank fusion.
- Identifier-aware tokenization and optional document/PDF/image loaders.
- Revision-aware exact caching and opt-in semantic cache evaluation.
- Segmented index lifecycle tooling and telemetry.
- Reproducible benchmark utilities.

Optional integrations are not required for core retrieval. This is not a full
replacement for a distributed vector database, an LLM framework or an access
control system. Confidence-based early exit remains opt-in with documented
quality trade-offs. Historical phase results are not new-release guarantees.

## Development

```sh
python -m pip install ".[dev]"
python -I -m pytest tests -q
```

The CI matrix covers Python 3.10–3.14 on Windows, Linux and macOS. A separate
Python 3.11 job runs historical evaluation-protocol tests with their optional
dependencies. A configured matrix is not evidence that every job passed.

[API reference](API.md) · [Changelog](CHANGELOG.md) ·
[Security](SECURITY.md) · [Historical development](docs/DEVELOPMENT_HISTORY.md)

Apache-2.0. See LICENSE, NOTICE and THIRD_PARTY.md.
