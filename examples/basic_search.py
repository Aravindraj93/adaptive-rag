"""Minimal domain-neutral retrieval example."""

from adaptive_rag import BM25Retriever, Chunk, Query, profile_hardware


chunks = [
    Chunk("returns", "policies", "Products may be returned within thirty days."),
    Chunk("shipping", "policies", "Standard shipping takes three to five business days."),
    Chunk("security", "help", "Enable two-factor authentication in account settings."),
]

retriever = BM25Retriever()
retriever.add(chunks)
results = retriever.search(Query("Products may be returned within how many days?"), top_k=2)

print(f"CPU threads: {profile_hardware().logical_cpus}")
for result in results:
    print(f"{result.rank}. {result.chunk.id} ({result.score:.3f}): {result.chunk.text}")

