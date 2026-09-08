# Security and operational boundaries

- Do not open arbitrary untrusted index files in a privileged process. Integrity
  checks, JSON limits, path checks and mapped-range checks reduce corruption risks;
  they do not sandbox hostile content or prevent every resource-exhaustion attack.
- Enforce authentication, tenant isolation and authorization in the application.
  Metadata filters and cache scopes are not access-control systems.
- Version caches with the corpus, model, tokenizer, retrieval settings and access
  policy. Never reuse generation tokens. Disable caching when state cannot be
  versioned reliably or the backend is nondeterministic/time-dependent.
- Cache query keys and result bodies are retained in process memory until eviction
  or clearing. Avoid secrets in telemetry attributes. Telemetry is bounded by
  default, but caller-provided attribute values can themselves be large.
- Serialize mutations or replace immutable backend snapshots. Do not assume a
  transaction spans lexical and dense indexes. Close readers before deleting their
  files, and use lease-aware segment cleanup.
- No remote listeners, credential collection, automatic uploads or LLM calls are
  included in this package.

No public vulnerability-reporting endpoint is configured yet. Share a minimal
reproduction privately with the project owner; do not post real confidential data.
