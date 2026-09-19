# Changelog

## 0.12.0 — domain adaptation & developer experience release

- Added high-level `HybridRetriever` facade combining lexical, dense, rank fusion,
  and revision-aware caching into a 3-line unified interface.
- Added `NormalizedTokenizer(split_identifiers=True)` for sub-token decomposition
  of snake_case, camelCase, and alphanumeric technical codes (`POPUP_PU1436`, `REQ_CFTS081`).
- Added `anchor_boost` and `anchor_gap_threshold` to `ReciprocalRankFusionRetriever`,
  locking exact identifier hits at Rank 1 to prevent MRR rank dilution from dense scores.
- Added `ScoreWeightedFusionRetriever` for normalized similarity-score fusion.
- Added `RelationalExpansionRetriever` for parent-child, trigger condition, and reference
  expansion across linked chunks.
- Added `FastPathIDLookupRetriever` for sub-millisecond direct exact code resolution.
- Added `MetadataScoreModifier` for source-type score boosting and history penalties.
- Added `CharNGramTokenizer` for character n-gram indexing of technical and automotive signals.
- Added optional NumPy SIMD dot-product acceleration in `DenseRetriever` with pure-Python
  fallback when NumPy is not installed (strictly preserving zero mandatory dependencies).
- Added `adaptive_rag.loaders` module (`TextLoader`, `JSONLoader`, `PDFLoader`, `ImageLoader`,
  `DirectoryLoader`, `load_file`) supporting PDF and image OCR via optional extras
  (`adaptive-rag[pdf]`, `adaptive-rag[images]`), custom OCR callbacks, and multimodal LLM vision callbacks.
- Added direct document, file, text, and directory ingestion methods to `HybridRetriever`
  (`add_text`, `add_file`, `add_directory`, `add_documents`).

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
