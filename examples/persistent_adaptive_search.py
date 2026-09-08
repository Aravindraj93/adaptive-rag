"""Chunk documents, persist BM25, reload it, and apply confidence gating."""

from pathlib import Path
from tempfile import TemporaryDirectory

from adaptive_rag import AdaptiveRetriever, BM25Retriever, Document, TokenChunker


documents = [
    Document("returns", "Products may be returned within thirty calendar days."),
    Document("shipping", "Standard shipping takes three to five business days."),
]

with TemporaryDirectory() as directory:
    index_path = Path(directory) / "policies.index.json"
    baseline = BM25Retriever()
    baseline.add_documents(documents, chunker=TokenChunker(chunk_size=20, overlap=2))
    baseline.save(index_path)

    retriever = AdaptiveRetriever(BM25Retriever.load(index_path))
    results = retriever.search("When can products be returned?", top_k=3)
    for result in results:
        print(f"{result.rank}. {result.chunk.document_id}: {result.chunk.text}")
    print(f"decision: {retriever.last_decision.reason}")
