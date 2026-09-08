"""Candidate-reusing fusion with an explicit illustrative policy.

Hashing embeddings and this threshold are demonstration choices, not semantic
quality claims. For real use, supply an embedder and a development-calibrated policy.
"""
from adaptive_rag import (AdaptiveFusionRetriever, BM25Retriever, Chunk, DenseRetriever,
                          FusionPolicy, HashingEmbedder, ReciprocalRankFusionRetriever)

chunks = [Chunk('a', 'a', 'Returns are accepted within thirty days.'),
          Chunk('b', 'b', 'Invoices are due at the end of each month.'),
          Chunk('c', 'c', 'The library opens at nine in the morning.')]
lexical = BM25Retriever()
dense = DenseRetriever(HashingEmbedder())
lexical.add(chunks)
dense.add(chunks)
fusion = ReciprocalRankFusionRetriever([lexical, dense])
retriever = AdaptiveFusionRetriever(fusion, policy=FusionPolicy(.8, (.5, .25, .25)))
for result in retriever.search('When can I return an item?', top_k=2):
    print(result.chunk.id, result.chunk.text)
print(retriever.last_decision)
