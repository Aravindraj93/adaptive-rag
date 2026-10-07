"""
Haystack 2.x integration example for adaptive-rag.

Shows how to use adaptive-rag as an AdaptiveHaystackRetriever component
with zero API cost, CPU-first execution, and built-in semantic caching.

Install:
    pip install adaptive-rag
"""

from adaptive_rag.integrations.haystack import AdaptiveHaystackRetriever

print("=" * 60)
print("Adaptive RAG Haystack 2.x Integration Example")
print("=" * 60)

# 1. Create the Haystack component from documents
retriever = AdaptiveHaystackRetriever.from_texts(
    texts=[
        "Refunds are available within thirty days of delivery.",
        "Standard ground delivery takes 3 to 5 business days.",
        "International shipping takes 7 to 14 business days.",
        "Customer support is reachable 24/7 at support@example.com.",
        "Subscriptions can be cancelled anytime without penalty.",
    ],
    scope="haystack-demo-app",
    top_k=3,
    use_cache=True,
)

# 2. Run retrieval using Haystack 2.x component interface (.run())
print("\nQuerying using Haystack component run(): 'cancelled subscriptions'")
output = retriever.run(query="cancelled subscriptions", top_k=2)

for i, doc in enumerate(output["documents"], 1):
    print(f"  [{i}] ID: {doc.id}")
    print(f"      Content: {doc.content}")
    print(f"      Meta: {doc.meta}")

# 3. In Haystack 2.x Pipelines, simply add as component:
print("\nHaystack Pipeline snippet:")
print("""
    from haystack import Pipeline
    from adaptive_rag.integrations.haystack import AdaptiveHaystackRetriever

    pipeline = Pipeline()
    pipeline.add_component("retriever", retriever)
    # pipeline.add_component("prompt_builder", prompt_builder)
    # pipeline.add_component("llm", open_ai_generator)
    # pipeline.connect("retriever", "prompt_builder.documents")

    # Run pipeline:
    # results = pipeline.run({"retriever": {"query": "refund policy"}})
""")

print("[OK] Haystack integration example completed successfully.")
