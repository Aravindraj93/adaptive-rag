# Getting Started with adaptive-rag

> **Goal:** Build a production-ready RAG system in 10 minutes, on CPU, with zero API costs.

## Prerequisites

```bash
pip install adaptive-rag
```

That's it. No GPU. No API key. No additional setup.

---

## Step 1: Index Your Documents

```python
from adaptive_rag import BM25Retriever, Chunk

# Create a retriever
retriever = BM25Retriever()

# Add your documents as chunks
retriever.add([
    Chunk(
        id="faq-1",           # Unique ID within your index
        source="faq.md",      # Document source (file path, URL, etc.)
        text="Refunds are available within 30 days of purchase.",
        metadata={"category": "returns", "priority": "high"},  # Optional
    ),
    Chunk("faq-2", "faq.md", "Shipping takes 3-5 business days."),
    Chunk("faq-3", "faq.md", "Express shipping takes 1 business day."),
    Chunk("faq-4", "faq.md", "Contact support at help@example.com."),
])

# Search
results = retriever.search("return policy", top_k=3)
for result in results:
    print(result.chunk.text)
```

**Output:**
```
Refunds are available within 30 days of purchase.
Shipping takes 3-5 business days.
Contact support at help@example.com.
```

---

## Step 2: Add Semantic Caching

Semantic caching makes repeated queries instant and eliminates API costs for cached results.

```python
from adaptive_rag import BM25Retriever, Chunk, SemanticCachedRetriever

backend = BM25Retriever()
backend.add([...])  # Your chunks

# The revision controls cache invalidation.
# Increment it when your corpus changes.
revision = 1

cached = SemanticCachedRetriever(
    backend,
    scope="my-app",           # Unique name for this cache
    revision=lambda: revision, # Called on each search
)

# First call: goes to BM25
result1 = cached.search("refund policy", top_k=3)

# Second call with identical query: instant from cache
result2 = cached.search("refund policy", top_k=3)

# When corpus changes, increment revision to invalidate
revision = 2
result3 = cached.search("refund policy", top_k=3)  # Fresh query

print(cached.info())
# {'hits': 1, 'misses': 2, 'scope': 'my-app', 'revision': 2}
```

> **When to increment revision:** Any time you add, remove, or update documents. Also when you change retrieval settings (top_k, embedder, etc.).

---

## Step 3: Load Documents from Files

Instead of manually creating Chunk objects, use the built-in loaders:

```python
from adaptive_rag import DirectoryLoader, BM25Retriever

# Load all text files from a directory
loader = DirectoryLoader("./my_docs/")
documents = loader.load()

# Convert to chunks and index
retriever = BM25Retriever()
retriever.add([
    chunk
    for doc in documents
    for chunk in doc.chunks  # Documents split into chunks automatically
])
```

**Or use the domain packs (recommended for most use cases):**

```python
from adaptive_rag.packs import HealthcareRAG, LegalRAG, EducationRAG

rag = HealthcareRAG()
rag.add_documents("./clinical_guidelines/")  # Handles loading + chunking
results = rag.search("hypertension treatment")
```

---

## Step 4: Use with LangChain or LlamaIndex

### LangChain

```python
from adaptive_rag.integrations.langchain import AdaptiveRetriever

retriever = AdaptiveRetriever.from_texts(
    texts=["doc1 text", "doc2 text"],
    scope="my-app",
)

# As a LangChain retriever
lc_retriever = retriever.as_langchain_retriever()

# In a chain
from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(
    llm=your_llm,
    retriever=lc_retriever,
)
answer = chain.invoke({"query": "What is the return policy?"})
```

### LlamaIndex

```python
from adaptive_rag.integrations.llamaindex import AdaptiveQueryEngine

engine = AdaptiveQueryEngine.from_documents(
    documents=your_llama_docs,
    scope="my-app",
    llm=your_llm,
)

response = engine.query("What is the pricing?")
print(response.response)
```

---

## Step 5: Evaluate Your Retrieval

Before deploying, always measure your retrieval quality:

```bash
# Quick evaluation with built-in benchmark
adaptive-rag-evaluate --repetitions 5 --output report.json

# With your own dataset
adaptive-rag-evaluate --dataset my_questions.json

# Full corpus audit
adaptive-rag doctor --corpus ./my_docs/ --queries ./questions.json
```

**Questions file format:**
```json
[
  {
    "query": "What is the refund period?",
    "relevant_ids": ["faq-1"]
  },
  {
    "query": "How long does shipping take?",
    "relevant_ids": ["faq-2", "faq-3"]
  }
]
```

---

## Step 6: Scale to Large Corpora

For corpora with 100K+ documents, use the disk-backed MMap index:

```python
from adaptive_rag import MMapBM25Retriever

# Build once, persist to disk
MMapBM25Retriever.build("./my_index/", all_chunks)

# Load on startup (survives process restarts)
with MMapBM25Retriever.open("./my_index/") as retriever:
    results = retriever.search("query", top_k=5)
```

For hybrid (BM25 + dense) at scale:

```python
from adaptive_rag import HybridRetriever

# Bring your own embedder — no model is downloaded automatically
# Options: sentence-transformers, OpenAI, Cohere, local GGUF, etc.
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")  # ~80MB, CPU-friendly

retriever = HybridRetriever(embedder=model.encode)
retriever.add(chunks)
results = retriever.search("complex semantic query", top_k=10)
```

---

## Tips & Best Practices

### Chunk Size

| Use Case | Recommended Size | Reason |
|---|---|---|
| FAQ / Q&A | 128–256 tokens | Short, precise answers |
| Product docs | 256–512 tokens | Context + precision balance |
| Legal / contracts | 512–1024 tokens | Must preserve clause context |
| Books / long docs | 256–512 tokens | Good recall without losing focus |

### Cache Scope

Use a unique `scope` per:
- Application / service
- User permission group (don't share cache across permission levels)
- Language (if multilingual)

### Revision Strategy

```python
# Immutable corpora (FAQ, policies) — use fixed revision
revision = hash(tuple(sorted(chunk.id for chunk in chunks)))

# Growing corpora — use document count or timestamp
revision = len(chunks)

# Time-sensitive corpora — use date
revision = datetime.date.today().isoformat()
```

### Performance Tips

1. Install numpy for SIMD acceleration:
   ```bash
   pip install adaptive-rag[numpy]
   ```

2. Use `MMapBM25Retriever` for large corpora (>50K chunks)

3. Set a reasonable `top_k` — retrieving 100 results when you need 5 is wasteful

4. Pre-warm the cache by running your most common queries on startup

---

## Next Steps

- [API Reference](../API.md) — Complete API documentation
- [Examples](../examples/) — More code examples
- [Benchmarks](../benchmarks/) — Benchmark datasets
- [Contributing](../CONTRIBUTING.md) — Help improve adaptive-rag

---

*adaptive-rag: CPU-first RAG for everyone. No GPU. No cloud. No API bill.*
