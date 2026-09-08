# Changelog

## 0.11.0 — consolidated experimental release

- Added bounded exact-result LRU caching with explicit scope/revision contracts,
  deep-copied cached results, single-flight serialized misses and diagnostics.
- Bounded telemetry to 1,024 events by default; explicit `None` retains old unlimited
  behavior. Added duration validation and dropped-event accounting.
- Rejected duplicate/nonfinite manifest JSON and oversized manifests. Validated
  mapped offsets, dense dimensions and sparse parameters; fixed partial-open cleanup.
- Preserved opt-in confidence routing and conservative always-fusion policy defaults.
- Added multi-domain ranking replay, real CPU repeated-query timing, concurrent
  serving/index lifecycle checks and compatibility verification tooling.
- Consolidated API/security/compatibility documentation, supplied a CI matrix,
  included complete Apache-2.0 license text and updated packaging metadata.

This is not a production-ready 1.0 release. Novel-query semantic routing still has
documented quality trade-offs. Earlier phase reports are historical evidence.
