"""adaptive-rag integrations package.

Provides drop-in adapters for popular AI/ML frameworks.

Available integrations:
    - adaptive_rag.integrations.langchain  — LangChain BaseRetriever adapter
    - adaptive_rag.integrations.llamaindex — LlamaIndex QueryEngine adapter

Example:
    # LangChain
    from adaptive_rag.integrations.langchain import AdaptiveRetriever
    retriever = AdaptiveRetriever.from_texts(texts, scope="my-app")
    lc_retriever = retriever.as_langchain_retriever()

    # LlamaIndex
    from adaptive_rag.integrations.llamaindex import AdaptiveQueryEngine
    engine = AdaptiveQueryEngine.from_documents(docs, scope="my-app")
"""

__all__ = [
    "AdaptiveRetriever",   # LangChain
    "AdaptiveQueryEngine", # LlamaIndex
]


def _lazy_import(name: str) -> object:
    """Provide a helpful error when an integration module is imported."""
    raise ImportError(
        f"Could not import {name}. "
        "Make sure you have the required framework installed:\n"
        "  LangChain:  pip install langchain-core\n"
        "  LlamaIndex: pip install llama-index-core"
    )
