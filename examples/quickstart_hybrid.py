"""Quickstart example demonstrating adaptive-rag 0.12.0 HybridRetriever."""

from adaptive_rag import Chunk, HybridRetriever


def simple_semantic_embedder(texts):
    """Deterministic 8-dimensional semantic concept projection for demonstration."""
    concepts = [
        "requirement", "specification", "popup", "interface",
        "signal", "network", "battery", "drive"
    ]
    vectors = []
    for text in texts:
        t_low = text.casefold()
        vec = [float(concept in t_low) for concept in concepts]
        # Avoid all-zero vector for demonstration
        if sum(vec) == 0:
            vec[0] = 0.1
        vectors.append(vec)
    return vectors


def main():
    print("--- 1. Initializing HybridRetriever ---")
    # HybridRetriever automatically configures BM25 (with identifier splitting),
    # Dense retrieval, anchor-boosted RRF, and optional revision-safe caching.
    retriever = HybridRetriever(
        embedder=simple_semantic_embedder,
        split_identifiers=True,
        anchor_boost=1.0,
        revision=lambda: "corpus-v1",
        scope="quickstart-demo",
    )

    print("--- 2. Indexing Chunks ---")
    chunks = [
        Chunk("req-081", "doc-1", "REQ_CFTS081: DriveMode controller parameter specification."),
        Chunk("pu-1436", "doc-2", "POPUP_PU1436: Confirmation dialog for DriveMode switch."),
        Chunk("sig-001", "doc-3", "CAN_MSG_DRV_MODE.SIG_STAT_REQ transmission frequency 100ms."),
        Chunk("gen-001", "doc-4", "General battery safety guide and high-voltage precautions."),
    ]
    retriever.add(chunks)
    print(f"Indexed {len(retriever)} chunks.")

    print("\n--- 3. Exact Code Search (Anchor Boost locks Rank 1) ---")
    for r in retriever.search("REQ_CFTS081", top_k=2):
        print(f"Rank {r.rank}: [{r.chunk.id}] (Score: {r.score:.4f}) - {r.chunk.text}")

    print("\n--- 4. Semantic / Natural Language Search ---")
    for r in retriever.search("battery safety precautions", top_k=2):
        print(f"Rank {r.rank}: [{r.chunk.id}] (Score: {r.score:.4f}) - {r.chunk.text}")

    print("\n--- 5. Cached Query Verification ---")
    cached_result = retriever.search("REQ_CFTS081", top_k=2)
    print(f"Repeat query result rank 1: [{cached_result[0].chunk.id}] (Instant exact reuse)")


if __name__ == "__main__":
    main()
