"""
adaptive-rag quickstart example.

This file demonstrates the most common usage patterns.
Run it with: python examples/quickstart/basic_usage.py
"""

from adaptive_rag import BM25Retriever, Chunk, SemanticCachedRetriever

# ─── 1. Basic BM25 retrieval ──────────────────────────────────────────────────
print("=" * 60)
print("1. Basic BM25 Retrieval (CPU-only, zero dependencies)")
print("=" * 60)

retriever = BM25Retriever()
retriever.add([
    Chunk("doc-1", "handbook", "Refunds are available within thirty days of purchase."),
    Chunk("doc-2", "handbook", "Standard shipping takes 3 to 5 business days."),
    Chunk("doc-3", "handbook", "Express shipping delivers within 1 business day."),
    Chunk("doc-4", "handbook", "Contact our support team at help@example.com."),
    Chunk("doc-5", "handbook", "We accept Visa, MasterCard, and PayPal payments."),
    Chunk("doc-6", "handbook", "Products can be exchanged within 60 days."),
])

results = retriever.search("how long do I have to return something", top_k=3)
print("\nQuery: 'how long do I have to return something'")
for i, result in enumerate(results, 1):
    print(f"  [{i}] (id={result.chunk.id}): {result.chunk.text}")


# ─── 2. Semantic caching ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Semantic Caching (repeated queries are free)")
print("=" * 60)

import time

revision = 1
cached_retriever = SemanticCachedRetriever(
    retriever,
    scope="quickstart-demo",
    revision=lambda: revision,
)

queries = [
    "refund period",
    "refund period",      # identical — should be cached
    "return window",      # semantically similar (exact match cache: may miss)
    "shipping time",
    "shipping time",      # cached
]

for query in queries:
    t0 = time.perf_counter()
    results = cached_retriever.search(query, top_k=2)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  Query: {query!r:30s} -> {elapsed:5.2f}ms | top result: {results[0].chunk.id if results else 'none'}")

info = cached_retriever.info()
print(f"\n  Cache stats: {info}")


# ─── 3. Hybrid retrieval (BM25 + dense) ──────────────────────────────────────
print("\n" + "=" * 60)
print("3. Hybrid Retrieval (BM25 + dense, BYOE*)")
print("   * Bring Your Own Embedder")
print("=" * 60)

try:
    from adaptive_rag import HybridRetriever, HashingEmbedder

    # HashingEmbedder is a simple hash-based embedder for demo purposes.
    # In production, replace with sentence-transformers or any callable:
    #   embedder = SentenceTransformer("all-MiniLM-L6-v2").encode
    hybrid = HybridRetriever(embedder=HashingEmbedder())
    hybrid.add([
        Chunk("doc-1", "handbook", "Refunds are available within thirty days of purchase."),
        Chunk("doc-2", "handbook", "Standard shipping takes 3 to 5 business days."),
        Chunk("doc-3", "handbook", "Express shipping delivers within 1 business day."),
        Chunk("doc-4", "handbook", "Contact our support team at help@example.com."),
    ])

    results = hybrid.search("money back guarantee", top_k=2)
    print(f"\n  Query: 'money back guarantee'")
    for i, r in enumerate(results, 1):
        print(f"    [{i}] {r.chunk.text}")

except ImportError as e:
    print(f"  Skipped (install numpy extra): {e}")


# ─── 4. Metadata filtering ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Metadata Filtering")
print("=" * 60)

from adaptive_rag import MetadataFilter, MetadataFilteredRetriever

base = BM25Retriever()
base.add([
    Chunk("hr-1",  "hr",   "Vacation policy: 20 days per year.", metadata={"dept": "hr"}),
    Chunk("hr-2",  "hr",   "Health insurance covers dental and vision.", metadata={"dept": "hr"}),
    Chunk("eng-1", "eng",  "On-call rotation is every 4th week.", metadata={"dept": "engineering"}),
    Chunk("eng-2", "eng",  "Code reviews must be approved by 2 engineers.", metadata={"dept": "engineering"}),
    Chunk("fin-1", "fin",  "Budget approvals over $10K need CFO sign-off.", metadata={"dept": "finance"}),
])

# Only return HR documents
hr_retriever = MetadataFilteredRetriever(base, MetadataFilter(equals={"dept": "hr"}))
results = hr_retriever.search("benefits coverage", top_k=5)
print(f"\n  Query: 'benefits coverage' | Filter: dept=hr")
for r in results:
    print(f"    [{r.chunk.id}] {r.chunk.text}")


# ─── 5. Run the evaluation CLI ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Evaluation CLI")
print("=" * 60)
print("""
  Run the built-in evaluation from the command line:

    adaptive-rag-evaluate --repetitions 5 --output report.json
    adaptive-rag-evaluate --dataset benchmarks/general_knowledge.json

  Or run the RAG Doctor to audit your corpus:

    adaptive-rag doctor --corpus ./my_docs/ --queries ./questions.json
""")

print("[OK] Quickstart complete! See README.md for more examples.")
