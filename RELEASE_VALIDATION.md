# 0.8.0rc1 validation

Validated locally on Windows, Python 3.11.9.

- Built both a source distribution and a wheel; the wheel was built from the sdist.
- Installed the wheel with `--no-deps` in a fresh virtual environment.
- Ran all 70 unit/integration tests using `python -I` against that installation.
- Ran all four example scripts using `python -I` against the installation.
- `pip check` reported no broken requirements.
- Tests include multiprocess writers, reader leases across processes, process
  termination, journal recovery, cleanup while readers remain open, corrupted
  manifests, filter parity, and rank parity between sparse backends.

Artifacts are under `dist/`. No registry publication was performed. Runtime
dependencies remain empty; ONNX Runtime, NumPy, and tokenizers were installed only
in a separate evaluation environment.

The build succeeded with setuptools deprecation warnings about the older license
metadata syntax. This syntax should be migrated before a stable release. No claim
of multi-platform validation or independent third-party audit is made.

Full public-corpus results and input provenance are in
`benchmark-phase8-cranfield.json`. Evaluation used 1,400 Cranfield documents and all
225 queries. RRF Recall@10 was 0.4324 versus 0.3916 for BM25. The default routed
pipeline escalated only three queries and provided little quality improvement.

Recommended status: controlled pilot with explicit retrieval-mode selection.
Default adaptive routing should be calibrated on separate development data before
it is trusted to trade retrieval quality against computation in deployment.

The historical README timings come from earlier phases. Phase 8's trained-model
retrieval timings exclude query embedding, which was cached for fair comparisons.
