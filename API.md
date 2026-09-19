# API contract for 0.12.x

The supported import surface is `adaptive_rag.__all__`. Private names, benchmark
helpers and source-layout paths are not a compatibility promise. Version 0.x remains
experimental; breaking behavior changes require a minor release and a changelog.
Patch releases should preserve public signatures and supported index schema.

`HybridRetriever(embedder, split_identifiers=True, anchor_boost=1.0, revision=None)`
provides a high-level facade orchestrating lexical BM25, dense semantic search,
anchor-boosted rank fusion, and optional revision caching through a simple `.add()`
and `.search()` interface.

All retrievers implement `search(query: str | Query, *, top_k: int = 5)` returning
ranked `SearchResult` objects. Concrete backends may additionally accept `where=`;
wrappers expose only arguments documented on their own signatures. Chunk IDs must
be unique within an index. Metadata used for persistence/filtering is JSON data.
Objects are shallowly frozen; applications should not mutate nested metadata.

`BM25Retriever.add(chunks)` builds in memory. `MMapBM25Retriever.build(path, chunks)`
creates an immutable disk index and returns an open reader. Use disk readers as
context managers or call `close()`. Segment managers append/delete/compact and keep
old-reader snapshots valid when reader-aware cleanup is used. Use v0.8+ readers
for cleanup; directly opened segment directories do not register leases.

`DenseRetriever(embedder)` accepts a callable mapping a sequence of texts to the
same number of finite nonzero vectors of one dimension. No model is selected or
downloaded automatically. Approximate LSH results can differ from exact dense search.
NumPy SIMD acceleration is automatically used when NumPy is installed in the runtime.

`ReciprocalRankFusionRetriever([lexical, dense], anchor_boost=0.0, anchor_gap_threshold=0.25)`
fuses backend rankings. Setting `anchor_boost > 0` preserves high-confidence exact identifier
hits from rank dilution by dense backends. `ScoreWeightedFusionRetriever([r1, r2], weights=...)`
fuses backends using normalized raw similarity scores. `RelationalExpansionRetriever(backend, chunk_store, relation_keys=...)`
expands search results by traversing entity and trigger references.

`NormalizedTokenizer(split_identifiers=True)` decomposes compound snake_case, camelCase,
and alphanumeric codes (e.g., `POPUP_PU1436` -> `popup`, `pu1436`, `pu`, `1436`).
`CharNGramTokenizer(min_n=3, max_n=4)` provides character n-gram indexing for cryptic technical symbols.

`FastPathIDLookupRetriever(fallback, id_index)` provides fast-path lookups for exact code
queries. `MetadataScoreModifier(retriever, multipliers=...)` applies multiplicative boosts
or penalties based on chunk metadata fields (such as source types).

An explicit `AdaptiveFusionRetriever(fusion, policy=FusionPolicy(...))` may skip other backends;
this can reduce relevance. `FusionPolicy()` selects always-fusion. `last_decision`
and similar diagnostics are not thread-local.

`CachedRetriever(backend, revision=callable, scope=string, max_entries=1024,
max_bytes=16777216)` adds bounded exact reuse. The revision must change for every
result-affecting state change and must not be recycled. `clear()` synchronizes cache
invalidation; `info()` returns counters. Cache counters are lifetime totals; clear
removes entries without resetting counts. Only strict JSON query metadata is keyed;
unsupported values bypass reuse. The cache does not forward backend-specific `where=`.
Use a configured filtering wrapper and revision that covers its filter settings.

`TelemetryCollector(max_events=1024)` retains a bounded event ring; `dropped_events`
counts overwritten events. `max_events=None` explicitly opts into unbounded history.
Snapshots are thread-safe, but event attribute values are not recursively immutable.

Persisted envelopes remain schema version 1. NaN/Infinity and duplicate JSON keys
are now rejected; manifests are limited to 128 MiB by the common reader. Checksums
are required by default. Range checks validate mapped offsets before use. Formats
are local trusted artifacts, not a hostile-input security boundary.
