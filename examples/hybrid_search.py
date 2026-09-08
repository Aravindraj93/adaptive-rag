"""Fuse lexical ranking with application-supplied semantic embeddings."""

from adaptive_rag import BM25Retriever, Chunk, DenseRetriever, ReciprocalRankFusionRetriever


def tiny_embedder(texts):
    concepts = ("return refund", "shipping delivery", "security password")
    return [
        [float(any(word in text.casefold() for word in concept.split())) for concept in concepts]
        for text in texts
    ]


chunks = [
    Chunk("returns", "policies", "Products may be returned within thirty days."),
    Chunk("shipping", "policies", "Standard delivery takes three to five days."),
    Chunk("security", "help", "Reset a forgotten password from security settings."),
]

lexical = BM25Retriever()
lexical.add(chunks)
semantic = DenseRetriever(tiny_embedder)
semantic.add(chunks)
retriever = ReciprocalRankFusionRetriever([lexical, semantic])

for result in retriever.search("refund policy", top_k=2):
    print(f"{result.rank}. {result.chunk.id}: {result.source}")
