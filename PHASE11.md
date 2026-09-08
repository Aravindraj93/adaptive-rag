# Phase 11 — conservative calibration and new-domain transfer

This is an evaluation-only phase. Runtime API, default policies, required
dependencies, and version 0.10.0rc1 are unchanged. Existing wheel/source archives
are the Phase 10 artifacts; new evaluator code and evidence live in this repository.

## Stricter selection

The 324 saved NFCorpus development observations are deterministically partitioned
by a salted SHA-256 query-ID ordering into 216 fitting queries and 108 guard queries.
This is **retrospective**: these observations were previously used in Phase 10.
The guard set is not a pristine holdout, and bootstrap results are diagnostic rather
than a generalization guarantee. No NFCorpus test observations enter this selection.

Fit uses the same fixed 315 selective policies plus always-fusion, but tightens the
allowable absolute Recall@10 and nDCG@10 loss from 0.01 to **0.005**. Only the
minimum-estimated-cost feasible proposal proceeds to the guard stage. Rejected
proposals are not replaced by trying additional policies on the guard observations.

For each quality metric, compute paired per-query differences against always-fusion
and 2,000 bootstrap means, with deterministic seed 1101. The lower 2.5th percentile
must be at least -0.01 for both metrics. Using two one-sided 97.5% estimates targets
a nominal 95% joint check by a Bonferroni argument, **subject to bootstrap validity**;
it is not an exact finite-sample coverage claim. Guard-estimated mean latency must
also improve by at least 5%. Otherwise explicitly select always-fusion.

The guard rejected the proposal with threshold 0.65 and weights (0.75, 0.25, 0):

| Guard metric | Mean difference vs fusion | Bootstrap lower estimate | Required minimum |
| --- | ---: | ---: | ---: |
| Recall@10 | -0.01575 | -0.03881 | -0.01000 |
| nDCG@10 | -0.01449 | -0.03964 | -0.01000 |

The frozen selection is therefore **always-fusion**, not a selective policy.
Its confidence gate does not run, and no query-embedding savings are expected.
This demonstrates rejection of unsupported optimization, not improvement in quality
or a successful adaptive-speed result. Costs used for calibration come from the
saved Phase 10 development measurements; they were not remeasured for this phase.

## Fresh transfer experiment

ArguAna is an argument-retrieval dataset, not a biomedical corpus. The archive and
published checksum come from the
[BEIR dataset catalog](https://github.com/beir-cellar/beir/wiki/Datasets-available).
We use the complete **8,674-document corpus** and a fixed **200-query sample from
1,406 queries**, not the full test set. The subset keeps this CPU experiment bounded.
IDs are ordered by SHA-256 of `phase11-sample:` plus query ID, and the first 200 are
selected. Policy and sample are saved before reading relevance judgments.

ArguAna includes query documents in its corpus. Identical query/document IDs must
not be returned; BEIR's
[evaluation implementation](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py)
also excludes identical IDs by default. Here, each backend requests one extra result,
removes a self-match, refills the requested depth and reranks before RRF. This avoids
letting the confidence gate accept a query's own document. It is explicitly our
pre-fusion exclusion protocol, not an assertion of exact leaderboard reproduction.

Pinned MiniLM assets, 256-token truncation, four CPU threads, masked-mean/L2 pooling,
BM25 defaults, and RRF settings are unchanged. Corpus embeddings may be cached;
query embeddings are computed live on every semantic search. Two timing repetitions
rotate mode order. Measurements cover warm search, including query embedding, but
exclude startup, indexing and download. Repetition does not double relevance sample
size. Pretrained model exposure to these public datasets is unknown.


## Transfer results

| Fixed 200-query ArguAna sample | Recall@10 | nDCG@10 | Mean search ms | p95 ms |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.7000 | 0.4102 | 227.99 | 317.84 |
| Always-fusion | 0.8100 | 0.5214 | 408.36 | 518.14 |
| Selected conservative policy (always-fusion) | 0.8100 | 0.5214 | 409.01 | 516.26 |

Fusion improved observed sample retrieval quality over BM25, at higher latency.
Selected-policy rankings are identical to always-fusion, with one live query
embedding call per search. Its small timing differences are measurement variation,
not adaptive overhead/savings evidence. There is **no demonstrated adaptive compute
saving** in this phase. These numbers cover the predeclared sample and protocol;
they are not full-dataset scores, statistical significance claims, or a deployment
recommendation. The 256-token embedding limit can truncate long arguments.

The final audit passed: exact calibration replay, policy/sample fingerprints,
self-match exclusion, 400 observations per mode across two repetitions, recomputed
metric means, and identical fusion/selected-policy rankings. The remaining 1,206
ArguAna queries were not evaluated; this sample is now observed.

## Evidence and reproduction

- [Frozen policy, full fitting candidates and guard check](phase11-results/phase11-policy.json)
- [Frozen query sample and input provenance](phase11-results/phase11-sample.json)
- [Transfer benchmark](phase11-results/benchmark-phase11-arguana.json)

Use the Phase 9 optional evaluation environment and pinned model assets. From the
repository root, with the public archive downloaded using normally validated HTTPS:

```powershell
python benchmarks/conservative_policy.py --development phase10-results/phase10-development.json --output NEW_RESULTS/phase11-policy.json
python benchmarks/evaluate_phase11.py --assets ASSET_DIRECTORY --dataset ARGUANA_ZIP --work CACHE_DIRECTORY --policy NEW_RESULTS/phase11-policy.json --output NEW_RESULTS
python benchmarks/verify_phase11.py NEW_RESULTS
python -m unittest discover -s benchmarks -p "test*protocol.py" -q
python -m unittest discover -s tests -q
```

Calibration refuses to overwrite its policy. Transfer evaluation refuses to replace
a completed report and checks any pre-existing sample against the supplied policy.
Invalid/nonfinite development metrics are rejected. The recorded policy can be
reproduced exactly from the saved development observations.

Validation: 80 existing library tests pass against the clean installed wheel, and
21 evaluation protocol tests pass, including 8 new checks for deterministic splits,
guard rejection, invalid data, bootstrap behavior and self-exclusion/reranking.

## Implications

Do not promote the Phase 10 fitted selective policy as a quality-preserving default.
The current development evidence supports always-fusion under the stricter gate.
The next useful input is broader representative, labeled calibration data across
domains, followed by a predeclared evaluation plan. More threshold experiments on
already-observed test results would not establish generalization.
