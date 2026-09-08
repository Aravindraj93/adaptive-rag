# Phase 10 — candidate-reusing adaptive fusion

Version 0.10.0rc1 adds an opt-in `AdaptiveFusionRetriever`, explicit `FusionPolicy`,
and `FusionDecision` diagnostics. Existing `EscalatingRetriever` and ordinary fusion
defaults remain unchanged. Required runtime dependencies remain empty.

## What changed

The new router searches the first fusion backend once, at the full fusion candidate
depth (`top_k * candidate_multiplier`). If it escalates, fusion reuses those exact
ranked candidates and searches only the remaining backends. Reusing only the final
top-10 candidates would alter RRF results; this implementation preserves the full
depth and is tested against ordinary fusion. Empty candidate lists are also reused.
Candidates are local to one search call, not cached across queries.

Confidence combines three generic, bounded features: top-result query-term coverage,
relative top-score gap, and normalized top-10 score concentration. Nonnegative
weights sum to one. These scores are heuristics, **not calibrated probabilities**.
Tokenizer injection is supported; the first backend is intended to be a lexical
retriever with nonnegative relevance scores. No application-specific vocabulary or
document type is built into the router.

```python
from adaptive_rag import AdaptiveFusionRetriever, FusionPolicy

# fusion is an existing ReciprocalRankFusionRetriever with lexical first.
# Supply weights and a threshold selected on representative development data.
policy = FusionPolicy(threshold=chosen_threshold, weights=chosen_weights)
retriever = AdaptiveFusionRetriever(fusion, policy=policy)
results = retriever.search(query, top_k=10)
print(retriever.last_decision)
```

`FusionPolicy()` explicitly selects always-fusion. There is no new universal
confidence threshold. The new wrapper does not impose the old hardware top-k cap;
callers control candidate depth. Query objects pass through unchanged. Diagnostic
`last_decision` is not thread-local, and the wrapper does not create a transactional
snapshot spanning independently updated backends. No new cross-query cache exists.

See [the runnable example](examples/adaptive_fusion.py). Its hashing embedder and
illustrative threshold demonstrate API behavior, not retrieval quality.

## Fresh evaluation protocol

The new experiment uses NFCorpus's official development/test split and its complete
3,633-document corpus. The archive comes from the
[official BEIR catalog](https://github.com/beir-cellar/beir/wiki/Datasets-available),
with the published MD5 checked and SHA-256 recorded. Test judgments are first read
only after the policy has been saved and fingerprinted. Development and test query
IDs must be disjoint. NFCorpus was not used in the earlier phases.

The development search space is fixed in advance: 15 nonnegative weight triples on
a 0.25 simplex grid, each with 21 thresholds from 0 to 1, plus always-fusion. Select
minimum estimated mean latency subject to both development Recall@10 and nDCG@10
being no more than 0.01 below always-fusion. Primary timings include confidence
feature extraction; escalation costs add only the measured remaining fusion work.
Full fusion is separately timed, alternating measurement order. Reused rankings
must equal full-fusion rankings on every development query.

The pinned MiniLM model, four CPU threads, 256-token limit, masked-mean pooling,
normalization, BM25 defaults, and RRF settings are unchanged from Phase 9. Model
packages remain evaluation-only dependencies. Corpus vectors can be cached; query
vectors cannot. Test queries are measured twice with rotating mode order. Timing
covers warm in-process search including live query embeddings, excluding model
startup, indexing and download. Repetitions do not increase relevance sample size.

All development observations, 316 candidate policies, frozen selection, and per-query
test timings are saved in [phase10-results](phase10-results/phase10-policy.json).
The policy is specific to the development corpus; it is not promoted to a library
default. Pretraining overlap is unknown, and no statistical significance or broad
production-readiness claim is made.


## Measured results and release decision

The frozen policy uses threshold **0.60**, with feature weights **(0.75, 0, 0.25)**.
It was chosen from 324 development queries. Its SHA-256 is
`9030b12d4a489ed3d0c6ab1b59686a8647e7b720ef478e2979f3a4b22a05531a`.

| NFCorpus test, 323 queries | Recall@10 | nDCG@10 | Mean ms | p95 ms | Embedding calls/query |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.14646 | 0.32237 | 5.80 | 17.65 | 0 |
| Always-fusion | 0.17324 | 0.34836 | 78.97 | 108.83 | 1 |
| Selected adaptive policy | 0.16640 | 0.33756 | 36.38 | 91.54 | 0.418 |

The selected policy reduced observed mean search latency by **53.9%** and query
embedding calls by **58.2%** versus always-fusion. Its Recall@10 loss was 0.00685,
inside the predeclared 0.01 tolerance. Its nDCG@10 loss was **0.01080**, narrowly
outside that tolerance. **The held-out quality target was not fully met.** Do not
describe this as quality-preserving optimization or promote this fitted policy
to a default. The test outcomes were not used to adjust weights or thresholds.

Candidate reuse itself is verified by backend call counts and ranking parity;
the measured total speed reduction combines reuse with skipping semantic search.
This experiment does not isolate the speed benefit of reuse alone. Mean, median,
and p95 timings are in the raw report; desktop timing variability remains.

Validation: **80 library tests and 13 evaluation-protocol tests pass**. Report audits
verify dev/test separation, frozen policy hash, 646 observations per mode, metric
means, live embedding calls, and reuse flags. The new example runs successfully.
The package is still an experimental release candidate, not a published release.

Next: use a stricter development quality margin and uncertainty-aware selection,
then freeze a new policy before testing on another untouched dataset. This test
split is now observed; reusing it to tune a replacement would invalidate a fresh
held-out claim. No user documents are required for this phase.

## Reproduce and verify

Use the optional evaluation dependencies and pinned model assets described in
[Phase 9](PHASE9.md). Download `nfcorpus.zip` from the catalog above with a normally
validating HTTPS client. From the repository root:

```powershell
python -m pip install -e .
python benchmarks/evaluate_phase10.py --assets ASSET_DIRECTORY --dataset NFCORPUS_ZIP --work CACHE_DIRECTORY --output NEW_REPORT_DIRECTORY
python benchmarks/verify_phase10.py NEW_REPORT_DIRECTORY
python -m unittest discover -s tests -q
python -m unittest discover -s benchmarks -p "test*protocol.py" -q
python examples/adaptive_fusion.py
```

Use a fresh output directory: the evaluator refuses to overwrite a frozen policy.
Do not retune on the reported test results. The model and datasets are not bundled
in the runtime wheel. Public benchmark use does not override original data licenses.
