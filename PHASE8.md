# Phase 8 — release candidate 0.8.0rc1

## Review and corrections

A fresh local code review found inconsistent BM25 tie ordering, missing/null and
numeric-type discrepancies between Python filters and SQLite facets, nonfinite
embedding acceptance, and incomplete resource cleanup on failed segment opening.
Regression tests accompany the fixes. This was an in-session review, not an external
third-party audit. Broader hardening of all manifest readers remains outstanding.

Filters now require the field to exist and compare canonical JSON encodings.
Consequently `1`, `1.0`, and `true` are distinct, consistently across both paths.
An absent field does not match `null`. This is an intentional behavior correction.

## Segment reclamation

```python
manager = SegmentedBM25Index(path)
candidates = manager.cleanup()               # dry run
removed = manager.cleanup(dry_run=False)      # reclaim only unpinned segments
```

Readers acquire process-owned leases while opening their snapshot. Cleanup holds
the writer lock and reader-registration guard, and preserves current manifest
segments and every live reader's segments. OS locks release after process exit;
stale leases can then be removed. Concurrent-process and forced-exit tests exercise
this protocol. Failed readers close every successfully opened segment.

All participating readers must run v0.8 or newer; older readers and directly opened
`MMapBM25Retriever` segment paths do not register leases. Do not run cleanup while
such readers are active. Network filesystem semantics are not validated. Cleanup
permanently removes only recognized superseded segment directories, not live data.
Dry runs can remove stale lease bookkeeping. Unpublished incomplete build directories
are conservatively left alone. Initial creation is still not a multi-writer API.

## Public evaluation

Full Cranfield collection: 1,400 scientific abstracts, 225 natural-language queries,
all relevance judgments, with positive grades treated as relevant. Corpus text is
title plus abstract. Query IDs follow sequential query ordinals as required by the
official loader. No parameters were tuned on these queries.

Model: `sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU ONNX Runtime, four inference threads,
256-wordpiece truncation, attention-mask mean pooling and L2 normalization.

| Mode | Recall@10 | MRR@10 | nDCG@10 | Mean retrieval ms |
|---|---:|---:|---:|---:|
| BM25 | 0.3916 | 0.5106 | 0.2920 | 10.28 |
| MiniLM exact dense | 0.4161 | 0.5380 | 0.3151 | 29.51 |
| Reciprocal-rank fusion | 0.4324 | 0.5717 | 0.3299 | 41.01 |
| Default routed pipeline | 0.3916 | 0.5110 | 0.2922 | 11.21 |

Fusion improved recall by about 4.1 percentage points over BM25 on this collection.
Default routing escalated only 3 of 225 queries and captured almost none of this
gain. The default confidence heuristic needs calibration before relying on adaptive
routing for quality. These results support a controlled pilot, not general readiness
claims. One historical scientific collection cannot establish cross-domain quality;
possible pretrained-model exposure has not been audited.

Retrieval times use cached query embeddings, excluding query model inference.
The report separately records embedding time for all corpus and query inputs.
This evaluates the library's actual Python dense and fusion paths, not a substitute
vectorized search implementation. Reports include per-query ranked IDs, grades-based
metrics, hardware, dependency versions, model revision and asset SHA-256 digests.

Sources:
- https://ir-datasets.com/cranfield.html
- https://raw.githubusercontent.com/allenai/ir_datasets/master/ir_datasets/datasets/cranfield.py
- https://ir.dcs.gla.ac.uk/resources/test_collections/cran/cran.tar.gz
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## Reproduction

Install evaluation-only dependencies `numpy`, `onnxruntime`, and `tokenizers` into
an isolated environment. Download the model and dataset URLs pinned in the report's
`provenance` object into an assets directory and save that object as
`provenance.json`. Do not redistribute the public corpus as library test fixtures.

```powershell
python benchmarks/evaluate_cranfield.py --assets PATH_TO_ASSETS --output results.json
```

The evaluator verifies asset hashes before running. Runtime package dependencies
remain empty. Unit tests run offline without the trained model.

## Remaining release gates

- Calibrate routing on a separate development collection and validate on held-out queries.
- Validate Linux/macOS behavior and more malformed/corrupt manifests.
- Audit embedding-model identity, bounded telemetry, and high-cardinality filters.
- Validate reader cleanup on intended deployment filesystems and workloads.

This is a local release candidate; nothing has been published to a package registry.
