# 0.17.0rc1 local validation — 2026-09-29

Status: release candidate only; no stable tag or PyPI publication.

## Checks executed

- Windows / Python 3.11.9: 137 source-package tests passed with pytest.
- Historical evaluation protocols: 21 unittest tests passed using their existing
  separate NumPy/ONNX/tokenizers environment.
- Wheel built from the source distribution using isolated build environments.
- Wheel installed with --no-deps into a fresh virtual environment.
- All 137 tests passed against the installed wheel with Python isolated mode,
  run from the repository root so fixture paths resolve correctly.
- Installed basic example and evaluation CLI succeeded; pip check passed.
- Wheel/source distribution metadata checks passed with twine.
- An initial installed test invocation from the workspace parent failed two
  fixture-path lookups. Re-running from the documented repository directory
  passed; no assertions were weakened.

## Six-query smoke comparison

Five passes, 30 searches per backend, top_k=3; deliberately 80% repeated calls.
This fixture is too small to support general performance or quality claims.

- No ranking changes or labelled quality regressions.
- Both configurations: recall@3=1.0 and binary nDCG@3=1.0.
- Cache: 6 backend misses, 24 exact hits; no semantic hits tested by this run.
- Installed run overall median: baseline 0.0239 ms, cache 0.0232 ms.
- Installed run p95: baseline 0.0452 ms, cache 0.1898 ms.
- First-pass median: baseline 0.0321 ms, cache 0.1064 ms.
- Repeated-pass median: baseline 0.0214 ms, cache 0.0212 ms.

An earlier source run showed a slightly slower cache median. At these tiny
durations, timing variation dominates. The meaningful conclusion is preserved
fixture rankings and reduced backend calls, NOT an established speedup.

## Not yet validated

The full new-candidate Python/OS CI matrix, real Requirement Reader integration,
semantic equivalence on held-out real embeddings, optional PDF/OCR environments,
large-corpus cache overhead and end-to-end dollar savings. Historical 0.11
reports do not certify this candidate. The deferred ZIP modules remain unshipped.
