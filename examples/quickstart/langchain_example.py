"""
LangChain integration example for adaptive-rag.

Shows how to use adaptive-rag as a drop-in retriever in LangChain
pipelines — no API keys, no GPU, works on CPU.

Install:
    pip install adaptive-rag langchain-core

For full RAG chains:
    pip install adaptive-rag langchain-core langchain-openai
"""

from __future__ import annotations

# ─── Example 1: Standalone retrieval (no LLM needed) ─────────────────────────

from adaptive_rag.integrations.langchain import AdaptiveRetriever

# From plain text strings — easiest way to start
retriever = AdaptiveRetriever.from_texts(
    texts=[
        "Refunds are available within thirty days of purchase.",
        "Standard shipping takes 3 to 5 business days.",
        "Express shipping delivers within 1 business day.",
        "Contact support at help@example.com.",
        "We accept Visa, MasterCard, and PayPal.",
        "Products can be exchanged within 60 days.",
    ],
    scope="my-ecommerce-app",
    use_cache=True,
)

# Retrieve directly (returns LangChain Documents)
docs = retriever.get_relevant_documents("refund policy")
print("Example 1: Standalone retrieval")
for doc in docs:
    print(f"  - {doc.page_content[:80]}")
    print(f"    metadata: {doc.metadata}")


# ─── Example 2: As a LangChain retriever in a chain ──────────────────────────

print("\nExample 2: With LangChain RetrievalQA chain")

# Uncomment when you have langchain-openai installed:
# from langchain_openai import ChatOpenAI
# from langchain.chains import RetrievalQA
#
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever.as_langchain_retriever(),
#     return_source_documents=True,
# )
# result = chain.invoke({"query": "What is the return policy?"})
# print(f"  Answer: {result['result']}")
# print(f"  Sources: {[d.metadata['chunk_id'] for d in result['source_documents']]}")

print("  -> Install langchain-openai and uncomment the code above")


# ─── Example 3: From existing LangChain documents ────────────────────────────

print("\nExample 3: From LangChain Document objects")

try:
    from langchain_core.documents import Document

    lc_documents = [
        Document(
            page_content="Our annual subscription costs $99 per user.",
            metadata={"source": "pricing.md", "department": "sales"},
        ),
        Document(
            page_content="Enterprise plans start at $499/month for 10 users.",
            metadata={"source": "pricing.md", "department": "sales"},
        ),
        Document(
            page_content="Free tier includes 5GB storage and 1000 API calls.",
            metadata={"source": "pricing.md", "department": "sales"},
        ),
    ]

    retriever2 = AdaptiveRetriever.from_langchain_docs(
        lc_documents,
        scope="pricing-docs",
        use_cache=True,
    )

    results = retriever2.get_relevant_documents("how much does the enterprise plan cost")
    for doc in results:
        print(f"  - {doc.page_content}")

except ImportError:
    print("  -> Install langchain-core: pip install langchain-core")


# ─── Example 4: Adding documents incrementally ────────────────────────────────

print("\nExample 4: Incremental document addition")

try:
    from langchain_core.documents import Document

    retriever3 = AdaptiveRetriever.from_texts(
        ["Initial document about our product."],
        scope="growing-corpus",
    )

    # Add more documents as they arrive
    new_docs = [
        Document(page_content="New feature: dark mode is now available.", metadata={"source": "changelog"}),
        Document(page_content="Bugfix: fixed login issue on mobile.", metadata={"source": "changelog"}),
    ]
    retriever3.add_documents(new_docs)

    results = retriever3.get_relevant_documents("dark mode feature")
    for doc in results[:2]:
        print(f"  - {doc.page_content}")

except ImportError:
    print("  -> Install langchain-core: pip install langchain-core")

print("\n[OK] LangChain integration examples complete!")
print("   See: https://github.com/Aravindraj93/adaptive-rag")
