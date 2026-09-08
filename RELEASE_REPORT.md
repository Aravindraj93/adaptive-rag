# 0.11.0 release completion report

This consolidated **experimental** release completes the local implementation,
validation and packaging work for the four requested workstreams. It does not turn
unsupported production claims into guarantees. No package has been published.

## 1. Reliable adaptive decisions

Implemented bounded exact-result caching with a mandatory scope/revision contract,
deep-copied cache payloads, query-context snapshots, eviction limits and serialized
single-flight misses. Repeat queries use identical results; misses use the full
configured retriever. Heuristic semantic routing remains opt-in because its quality
preservation on novel queries is not established. The stricter calibration gate
continues to reject the unsupported selective proposal.

On 20 previously observed NFCorpus queries repeated five times, using the full
3,633-document corpus and real CPU MiniLM inference:

| Strategy | Mean search latency | Query embedding calls | Searches |
| --- | ---: | ---: | ---: |
| Full fusion | 87.39 ms | 100 | 100 |
| Exact revision-aware cache | 17.30 ms | 20 | 100 |

All 100 result sets matched exactly. The designed repeat rate was 80%; observed
mean latency reduction was about 80%. This is **repeat-workload evidence**, not
novel-query savings or a universal performance guarantee. The implementation must
be used with reliable, never-recycled revisions and coordinated immutable state.

## 2. Representative calibration and evaluation

Preserved prior public evaluations and their negative findings; no test set was
retuned to manufacture success. Replayed 973 previously observed query rankings
across Cranfield, SciFact, NFCorpus and an ArguAna sample, with three requests per
query: all 2,919 rankings were unchanged under exact caching. This is correctness
replay, not fresh relevance or model-latency evaluation. Prior guard/test limitations
remain documented in PHASE9–PHASE11. Broader application-specific calibration is
still required before promoting semantic shortcuts in a deployment.

## 3. Production hardening work

- Default telemetry now retains 1,024 events; unlimited retention requires opt-in.
- Strict manifest parsing rejects duplicate keys, nonfinite data and oversized
  envelopes. Mapped offsets, dimensions and sparse parameters are checked.
- Partial-open dense-reader failure now closes the first file correctly.
- 97 library tests pass, including cache/context/invalidation/concurrency tests,
  corruption checks, process-lock recovery and reader-safe reclamation.
- 21 separate evaluation protocol tests pass.
- 20,000 searches with eight workers over 10,000 synthetic documents preserved
  expected rankings; bounded cache behavior was checked.
- 30 append/delete/compact/cleanup cycles preserved live reader snapshots.
- A 100,000-document synthetic benchmark returned the expected top result on all
  100 queries for both in-memory and memory-mapped sparse retrieval. Retained traced
  Python allocations were approximately 231 MB vs 91 MB; these are not process RSS.
  Synthetic identifier queries are not representative general-search latency claims.

Platform/version clean-install outcomes are recorded in `release/compatibility.json`.
Clean wheel installs, all 97 library tests, all supplied examples, and dependency
checks passed on Windows with CPython **3.10.20, 3.11.9, 3.12.13, 3.13.14 and 3.14.0**.
The 21 optional evaluation tests were run separately on the evaluation environment.
Windows is the locally testable OS. A Windows/Linux/macOS CI matrix is supplied;
Linux and macOS execution is **not verified** here. No days-long soak, independent
security audit, network-filesystem or power-loss certification has been performed.

## 4. Release readiness

Feature/API scope frozen for 0.11.x; API, compatibility, security, changelog and
external-asset attribution documents supplied. The complete Apache-2.0 license and
modern license metadata are included. Runtime dependencies remain empty. Wheels
exclude model weights and external corpora. The source archive includes test and
benchmark tooling. There is no registry upload, public repository creation, or
claim that the package name is reserved.

Machine-readable evidence lives in `release/`: reliability replay/stress results,
live cache timing, 100k scale metrics, compatibility results and artifact hashes.
The final summary accompanying this report states the completed clean-install and
artifact checks. Earlier `dist/` files are historical; use version 0.11.0 artifacts.

## Remaining qualification limits

The **local experimental release** is the completion boundary. Production
qualification is not fully complete: novel-query routing quality, Linux/macOS
execution, sustained deployment behavior, external security review and
application-specific authorization/revision integration remain open. Exact caching
is useful now under its contract; it does not solve those separate questions.
