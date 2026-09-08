# Phase 7 — reliability (0.7.0)

Segment append, delete, compaction, and recovery now acquire an operating-system
writer lock with a 10-second timeout. Separate coordinator instances and processes
using these APIs share the same lock. The OS releases ownership if a process exits.
Initial creation and the separate consolidated `MMapIndexManager` are not covered.

Publication writes a checksummed `pending.json`, then atomically replaces
`segments.json`. Manifest writes use unique temporary files, flush and fsync them,
and replace the destination. `SegmentedBM25Index(path).recover()` validates and
finishes pending publication; every mutation also calls recovery under its lock.
These guarantees cover process interruption, not arbitrary power loss or network
filesystem semantics. An interrupted caller may have committed: recover and inspect
live IDs before retrying an append.

Compaction publishes a new segment within the same directory. Existing readers keep
their original snapshot; new readers see the new manifest. Superseded and orphaned
segments are retained deliberately, so disk usage grows until offline cleanup.
Automatic garbage collection and reader leases remain future work. Compaction
changes segment-local ranking as before; it does not promise identical ranking.

Deleted leading candidates no longer hide live matches in segmented search.
Older sparse indexes without a facet database reopen with a metadata scan fallback.

## Persist LSH

```python
index.save("lsh.json")
restored = LSHDenseRetriever.load("lsh.json", embedder=embedder)
restored.add(new_chunks)
```

The manifest stores normalized vectors, chunks, configuration and actual projection
planes. Loading validates dimensions, finite values and duplicate IDs, then rebuilds
buckets without re-embedding. Supply the same embedding model/configuration at load;
model identity is not verified automatically. Failed vector batches are validated
before modifying the index. Storage is JSON and rebuilds buckets at startup.

## Validation

The suite exercises competing writer processes, OS lock release after forced process
termination, simulated interruption between journal and manifest writes, readers
surviving compaction, deleted-head retrieval, and LSH round-trip/incremental behavior.
Phase 7 benchmark output is in `benchmark-phase7.json`; the README's Phase 6 timing
tables are historical measurements.

Binary offset tables, facet allowlists, trained embedding adapters and harder public
evaluation collections remain deferred. The next priority is reader-aware segment
garbage collection and stronger malformed-index validation.
