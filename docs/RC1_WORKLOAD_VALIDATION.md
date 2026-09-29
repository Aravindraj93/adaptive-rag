# Follow-up candidate validation — 2026-09-29

## Cross-platform CI

Commit `61d4ee8fda018cb0d608292c90c2ffe37c595985` passed all 16 jobs:
15 Python 3.10–3.14 / Windows, Linux, macOS combinations, plus evaluation protocols.
[Verified run](https://github.com/Aravindraj93/adaptive-rag/actions/runs/36588384265).
This is evidence for that commit, not for untested future changes.

Six subsequent safety tests also passed locally (143 library tests total):
exact-mode hard negatives, shadow-mode disagreement, TTL expiry during embedding,
dimension mismatches, concurrent tenant isolation, and public export consistency.
No production implementation changed in this follow-up.

## Synthetic workload experiment

10,000 chunks, 100 distinct query strings. Selective queries contain unique item
tokens. Broad queries also contain common tokens matching every chunk. Repeated
workloads replay five passes (500 requests, intentionally 80% repeat rate).
Two independent BM25 backends share identical corpus content; only the candidate
uses exact caching. This does not test semantic caching or LLM cost savings.

| Workload | Requests per backend | Baseline median ms | Cached median ms | Baseline p95 ms | Cached p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unique selective | 100 | 0.0121 | 0.0546 | 0.0249 | 0.1307 |
| Unique broad | 100 | 100.6401 | 100.5062 | 111.4970 | 117.9493 |
| Repeated selective | 500 | 0.0150 | 0.0357 | 0.0259 | 0.0812 |
| Repeated broad | 500 | 106.5643 | 0.1400 | 135.9603 | 113.4064 |

All four workloads had zero ranking changes and zero labelled regressions.
Recall@5 and binary nDCG@5 were 1.0 on these deliberately easy relevance labels.
Both repeated workloads used 100 backend searches instead of 500. Unique-query
workloads saved no searches. Tail latency includes misses: a low median does not
mean all requests are fast. The measured crossover depends on backend cost,
cache size, returned result size, repeat rate and hardware.

These are single-run illustrative measurements, not controlled performance
claims. Other local validation briefly overlapped execution. Synthetic labels
do not establish general retrieval quality. No process memory, dollar costs,
index-build time or embedding model quality was measured.

Reproduce from an installed checkout:

```sh
python -I benchmarks/evaluate_cache_workloads.py --chunks 10000 --queries 100 --output cache-workloads.json
```

Raw observations are in `release/cache-workloads-017rc1.json`.

## Remaining release gate

Requirement Reader integration requires its retrieval code and approved test
data. We need representative queries, chunk IDs/relevance judgments, the current
retriever configuration and an allowed execution environment. Do not share
confidential office material publicly. Without these inputs, only synthetic and
public-data validation can be completed; no office-quality claim is justified.
