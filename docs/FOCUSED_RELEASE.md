# Focused 0.17.0rc1 candidate

## Scope

Based on upstream commit f6d5558. Preserves the 0.12.0 public API and adds a
bounded cache wrapper plus a paired evaluator. Does not import the uploaded
ZIP's incomplete vision, graph, Hub, framework/server, uncertainty, multilingual,
streaming, async, expansion, observability or reranking implementations.
The ZIP is preserved separately. Its 123 failing tests have not been relabelled
as fixed: those modules are deferred, not part of this candidate.

The new cache intentionally replaces the unreleased ZIP API rather than exposing
its unsafe SemanticCache / CacheStatistics interfaces.

## Cache modes

```python
from adaptive_rag import SemanticCachedRetriever

# backend implements search(query, top_k=..., **supported_arguments).
# embed_query returns one deterministic, finite, nonzero vector.
shadow = SemanticCachedRetriever(
    backend, scope="application-A", revision=current_revision,
    mode="shadow", embedder=embed_query, threshold=0.95,
)
```

Use a dedicated instance per backend. Revision tokens must be monotonically
changed and never recycled. They cover the corpus, embedding model/version,
tokenizer, access policy and retrieval configuration. Configuration is fixed
for the instance; construct a new wrapper when changing cache mode/embedding.
Changing revisions is not a substitute for coordinating concurrent index writes.

Keys include scope, typed revision, top_k, Query.metadata and all backend kwargs.
String queries and Query objects remain separate. Unsupported non-JSON contexts
bypass caching. Caller arguments are forwarded to the backend: filters are not
implemented by the cache, and unsupported backend kwargs raise normally.

Entries expire after 300 seconds by default and are bounded by count and
serialized bytes. This is not a strict Python heap bound. Searches serialize
within one instance. Do not recursively call the same wrapper from its backend
or callbacks. No cross-process sharing or persistence is provided.

Exact hits do not call the embedder. On an exact miss, shadow/semantic modes
embed once and scan context-matching cached vectors linearly. This can be slower
than direct retrieval. Invalid embeddings fail explicitly rather than producing
misleading similarity values. Embedding dimensional mismatches do not match.

Shadow mode always runs the backend on an exact miss. Its counters compare
ordered chunk IDs only. They do not measure false-hit correctness; use relevance
labels and hard negatives to judge that. Semantic serving requires an explicit
reuse_validator(old_text, new_text) returning True. A validator approving every
pair is unsafe; it is an application contract, not a built-in equivalence proof.
Even with a validator, approximate hits return the earlier query's scores and
ordering. Keep exact mode where that behavior is unacceptable.

## Evaluate an existing application

```python
from adaptive_rag import ComparisonCase, compare_retrievers

cases = [ComparisonCase("refund period", frozenset({"refund-policy"}))]
report = compare_retrievers(baseline, candidate, cases, top_k=5, repetitions=1)
```

Use independent backends with identical corpora, models and retrieval settings.
Capture representative queries with permission; do not publish office documents.
Use separate tuning and held-out queries. Include never-repeated, exactly
repeated, paraphrased, near-but-different, filtered and post-update workloads.
Measure realistic repeat rates, not just deliberately repetitive smoke tests.

The evaluator alternates execution order, records each result and reports binary
recall/nDCG regressions. Empty/unavailable relevance labels are excluded and
reported as null, not proof of preserved quality. It does not measure index
builds, RAM, dollar cost or embedding calls. Instrument those separately for
end-to-end claims. Chunk IDs in reports may themselves be sensitive.

## Before a stable release

- Pass the installed-package CI matrix and distribution checks.
- Validate real office integration without leaking private data.
- Publish representative held-out quality/performance results, including losses.
- Validate semantic false hits on identifiers, numbers, negation and updates.
- Confirm package ownership/authentication before any PyPI publication.

No guaranteed cost savings, semantic correctness, or adoption targets are claimed.
