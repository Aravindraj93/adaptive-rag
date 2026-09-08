# Phase 9 — routing calibration and cross-domain evaluation

This phase adds an evaluation protocol, not a new runtime API. The library remains
0.8.0rc1; no defaults, required dependencies, integrations, or published packages
were changed. The existing `dist/` artifacts are the Phase 8 build.

## Protocol

The policy is selected using 75 Cranfield development queries (seed 42). The other
150 queries are held out of selection. This is a **retrospective split**: Phase 8
already reported full-Cranfield aggregate results. It is not a pristine unseen test.
The complete 1,400-document corpus remains available for every query. Cranfield
query IDs are sequential ordinals, as required by its relevance judgments.

Candidates are confidence thresholds 0 through 1 in steps of 0.05, plus an explicit
always-fusion policy. We minimize estimated mean development latency subject to
Recall@10 and nDCG@10 each being within 0.01 absolute of always-fusion. This is a
development quality constraint, not a held-out guarantee or significance test.
Escalation cost includes primary retrieval plus the full fusion call; the existing
router repeats lexical retrieval inside fusion. Empty primary results escalate.
Confidence equal to the threshold is accepted.

The selected policy is saved and SHA-256 fingerprinted **before** held-out evaluation.
It is then applied unchanged to all 300 SciFact test queries over 5,183 documents.
No SciFact training queries or judgments influence selection. The official
[BEIR dataset catalog](https://github.com/beir-cellar/beir/wiki/Datasets-available)
provides the archive and checksum. Its published MD5 is verified before use;
reports also record SHA-256. Archive members are read directly without extraction.
This is cross-domain evaluation, not proof the pretrained model never saw the data.

Retrieval settings are fixed: title plus abstract/text, BM25 defaults, exact dense
cosine search, equal-weight reciprocal-rank fusion with rank constant 60 and 30
candidates per backend for top-10 output. Graded nDCG uses exponential gain;
positive grades count toward recall. No reranker, LLM, or external vector database
is used.

Embeddings use pinned `sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, ONNX CPU execution with four intra-op
threads, 256-token truncation, attention-mask mean pooling, and L2 normalization.
Corpus embeddings are cached by asset/configuration/corpus fingerprint. **Query
embeddings are never cached:** every semantic search tokenizes and embeds its query.

Timings cover warm, in-process search end to end, including query embedding and
retrieval; they exclude model loading, corpus indexing, network requests, and process
startup. Each test query runs twice, rotating mode order. Reports include mean,
median, nearest-rank p95, per-query results, and actual embedding-call counts.
Repeated timings are not independent additional relevance samples. Desktop timing
noise remains possible; these are not production latency guarantees.

## Calibration outcome

**The selected policy is always-fusion (`threshold: null`).** The default confidence
heuristic did not demonstrate a cheaper policy within both development quality
constraints. The cheapest feasible threshold, 0.70, had estimated mean cost 90.35 ms;
always-fusion was 75.19 ms. The threshold-0.60 candidate was slightly faster at
74.14 ms but missed the recall floor. This outcome does not establish that all
adaptive routing approaches fail; it rejects this threshold gate under this protocol.

The policy fingerprint is
`7045016a3ba2e5fbd05daed3cd9bd8fb91a13dda4febb6150fb378d0d9977084`.
The `calibrated` mode therefore executes exactly the same retrieval as `fusion`.
Any timing difference between them is measurement variation, not routing savings.

## Results

See the machine-readable reports for every query and timing:

- [Frozen development policy and all threshold candidates](phase9-policy.json)
- [Cranfield held-out results](benchmark-phase9-cranfield-heldout.json)
- [SciFact transfer results](benchmark-phase9-scifact-transfer.json)

On Cranfield held-out queries, BM25 achieved Recall@10 0.3992 and nDCG@10 0.2972
at 16.52 ms mean. Fusion achieved 0.4228 and 0.3286 at 70.82 ms mean, with p95
87.14 ms. The selected policy produced identical rankings with one live embedding
call per query. It offers no demonstrated compute savings.

| Evaluation | Strategy | Recall@10 | nDCG@10 | Mean search ms | p95 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Cranfield held-out (150 queries) | BM25 | 0.3992 | 0.2972 | 16.52 | 25.65 |
| Cranfield held-out | Fusion | 0.4228 | 0.3286 | 70.82 | 87.14 |
| Cranfield held-out | Selected policy (fusion) | 0.4228 | 0.3286 | 71.28 | 84.91 |
| SciFact transfer (300 queries) | BM25 | 0.8093 | 0.6752 | 34.36 | 62.41 |
| SciFact transfer | Fusion | 0.8512 | 0.7069 | 147.85 | 251.43 |
| SciFact transfer | Selected policy (fusion) | 0.8512 | 0.7069 | 149.62 | 253.07 |

Fusion improved observed mean recall and nDCG over BM25 on both datasets, at
higher CPU latency. This supports a quality/cost trade-off, not a claim that the
adaptive optimization objective has been achieved. With always-fusion selected,
the transfer test evaluates that fixed strategy, not a successful selective gate.
No confidence intervals or statistical significance claims are made.

## Reproduce

Use Python 3.11 and install the library locally. Model dependencies are optional,
evaluation-only dependencies; the library itself still has zero required packages.
The measured environment used numpy 2.4.6, onnxruntime 1.29.0, and tokenizers 0.23.2.
Download the three pinned assets and `provenance.json` described in Phase 8, then run
from the repository root (replace the placeholder directories):

```powershell
python -m pip install -e .
python -m pip install numpy==2.4.6 onnxruntime==1.29.0 tokenizers==0.23.2
python benchmarks/evaluate_routing.py --assets ASSET_DIRECTORY --work CACHE_DIRECTORY --output REPORT_DIRECTORY
python -m unittest discover -s benchmarks -p test_routing_protocol.py -v
python benchmarks/verify_phase9.py REPORT_DIRECTORY
python -m unittest discover -s tests -q
```

The evaluator downloads the public SciFact archive if missing. If Python cannot
validate that host's certificate, download the same URL using a normally validating
system client into `CACHE_DIRECTORY/scifact.zip`; do not disable TLS verification.
That was necessary on this Windows machine. Use `--resume-frozen` after an interrupted
download to retain the original policy and Cranfield report; their hashes must match.

The first run retained aggregate development candidate statistics but not per-query
development observations. The final evaluator additionally writes
`phase9-development.json` on fresh runs. That file is deliberately not reconstructed
for the recorded experiment, because new timing observations could change selection.
Re-running into a separate report directory preserves the published experiment.

Validation: 70 existing library tests and 8 new evaluation protocol tests pass.
The saved-report audit also passes: disjoint development/held-out IDs, policy
fingerprints, query/repetition counts, recomputed metric means, unique top-10 IDs,
identical fusion/selected-policy rankings, and live embedding-call counts.
The protocol tests cover quality-preserving primary selection, escalation cost,
always-fusion selection, empty candidates, boundary equality, and graded metrics.

## Next step

Improve the router before claiming adaptive savings: reuse sparse candidates during
fusion escalation and evaluate richer, generic confidence features on development
data. Freeze the new policy, then use a new untouched evaluation set. Both current
test sets have now been observed. User-provided representative queries and relevance
labels would be useful later, but are not required to finish this phase.
