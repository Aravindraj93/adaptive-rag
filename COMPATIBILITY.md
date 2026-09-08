# Compatibility and deployment limits

Python syntax/API target: CPython 3.10–3.14. Verified versions and operating systems
are recorded in `release/compatibility.json`, not inferred from classifiers.
The supplied CI matrix covers Windows, Linux and macOS, but an unexecuted job is
not a passing result. This machine has no Linux runtime and cannot run macOS tests.

Disk formats use schema version 1. Keep original indexes/backups before upgrades.
The 0.11 reader rejects malformed/nonfinite manifests that older code may have
accepted. A checksum verifies integrity, not authenticity. No power-loss or network
filesystem durability guarantee is made. Atomic replacement is not a distributed
transaction. Use local filesystems and coordinate independent backend mutations.

Segmented search fuses segment-local rankings and is not guaranteed to match one
global BM25 index. Compaction can change rankings. Index reclamation requires the
reader lease protocol; stale process recovery is tested locally, not on every OS.

Exact dense search is linear in corpus size and dimensions. Python CPU performance
may be unsuitable for large low-latency production workloads. LSH is approximate.
Hardware budgets and confidence scores are heuristics, not calibrated probabilities.

Release validation is bounded: it includes a finite concurrent-serving stress run,
index lifecycle cycles, scale measurements and clean installs. It is not a days-long
soak, third-party security audit, or production certification. Retrieval datasets do
not establish relevance for every application. Semantic-routing quality preservation
on novel queries remains unproven; it is not a default release promise.

The package name has not been reserved or checked for publishing availability.
Nothing is uploaded to PyPI/GitHub by the release preparation process.
