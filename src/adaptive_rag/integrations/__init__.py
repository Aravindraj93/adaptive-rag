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

from .haystack import AdaptiveHaystackRetriever
from .langchain import AdaptiveRetriever, AdaptiveRetriever as LangChainAdaptiveRetriever
from .llamaindex import AdaptiveBaseRetriever, AdaptiveQueryEngine

__all__ = [
    "AdaptiveRetriever",
    "LangChainAdaptiveRetriever",
    "AdaptiveQueryEngine",
    "AdaptiveBaseRetriever",
    "AdaptiveHaystackRetriever",
]
