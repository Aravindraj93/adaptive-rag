<div align="center">

# 🧠 adaptive-rag

**Smart RAG retrieval for everyone. No GPU. No embedding API. No cloud bill.**

Run production-quality retrieval on any laptop or server.  
Drop into your existing LangChain / LlamaIndex app in 3 lines.

[![PyPI](https://img.shields.io/pypi/v/adaptive-rag?color=blue&label=PyPI)](https://pypi.org/project/adaptive-rag)
[![Python](https://img.shields.io/pypi/pyversions/adaptive-rag)](https://pypi.org/project/adaptive-rag)
[![License](https://img.shields.io/github/license/Aravindraj93/adaptive-rag)](LICENSE)
[![CI](https://github.com/Aravindraj93/adaptive-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Aravindraj93/adaptive-rag/actions)
[![CPU Only](https://img.shields.io/badge/runs%20on-CPU%20only-brightgreen)]()
[![Zero Deps](https://img.shields.io/badge/mandatory%20deps-zero-brightgreen)]()
[![Stars](https://img.shields.io/github/stars/Aravindraj93/adaptive-rag?style=social)](https://github.com/Aravindraj93/adaptive-rag/stargazers)

</div>

---

## Why adaptive-rag?

Most RAG systems have a dirty secret: **they're expensive and fragile in production**.

| | Typical RAG Stack | **adaptive-rag** |
|---|:---:|:---:|
| GPU required | ✅ Often | ❌ Never |
| Mandatory embedding API | ✅ Yes (\$50–\$800/month) | ❌ No (\$0) |
| Built-in evaluation CLI | ❌ No | ✅ Yes |
| Semantic caching | ❌ No | ✅ Yes |
| Zero mandatory dependencies | ❌ No | ✅ Yes |
| HIPAA/on-prem safe | ❌ No | ✅ Yes |
| Works on a Raspberry Pi | ❌ No | ✅ Yes |

**adaptive-rag is the retrieval layer that works for everyone** — from a startup's first product to a hospital's air-gapped server.

---

## ⚡ 30-Second Quickstart

```bash
pip install adaptive-rag
```

```python
from adaptive_rag import BM25Retriever, Chunk, SemanticCachedRetriever

# 1. Create a retriever — no model downloads, no API keys
backend = BM25Retriever()
backend.add([
    Chunk("policy-1", "docs", "Refunds are available within thirty days."),
    Chunk("policy-2", "docs", "Shipping takes 3-5 business days."),
    Chunk("policy-3", "docs", "Contact support at help@example.com."),
])

# 2. Wrap with semantic caching — repeated queries are free
retriever = SemanticCachedRetriever(
    backend, scope="public-docs", revision=lambda: 1
)

# 3. Search — CPU only, milliseconds latency
results = retriever.search("refund period", top_k=3)
for r in results:
    print(r.chunk.text)
```

---

## 🩺 RAG Doctor — Diagnose Your Retrieval in Seconds

```bash
# Diagnose any existing RAG corpus
adaptive-rag doctor --corpus ./my_docs/ --queries ./questions.json

# Or run the built-in evaluation benchmark
adaptive-rag-evaluate --repetitions 5 --output report.json
```

**Sample output:**
```
╔══════════════════════════════════════════════════════════╗
║         📊 RAG Health Report — my_docs/                  ║
╠══════════════════════════════════════════════════════════╣
║  Retrieval Quality:    B+  (82/100)                      ║
║  Recall@5:             0.847                             ║
║  MRR:                  0.791                             ║
║  Cache Hit Rate:       0% → 73% (with adaptive-rag)      ║
║  Estimated API Savings: ~$340/month                      ║
╠══════════════════════════════════════════════════════════╣
║  💡 Top Recommendations:                                 ║
║  1. Enable SemanticCachedRetriever (saves 73% API calls) ║
║  2. Use HybridRetriever for 12% recall improvement       ║
║  3. Adjust chunk size to 512 tokens (currently 1024)     ║
╚══════════════════════════════════════════════════════════╝
```

---

## 🌐 HTTP REST Server (TypeScript / Frontend / Microservices)

Serve your local document index over HTTP on CPU with zero dependencies:

```bash
# Start server
adaptive-rag serve --corpus ./my_docs/ --port 8000
```

Query it using standard `curl` or any language:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "refund policy", "top_k": 3}'
```

Or use the zero-dependency JavaScript/TypeScript client ([`client-js/`](client-js/)):

```typescript
import { AdaptiveRAGClient } from "./client-js"; // or from npm package

const client = new AdaptiveRAGClient({ baseUrl: "http://127.0.0.1:8000" });
const results = await client.search("refund policy", 3);
console.log(results);
```

---

## 🔌 Framework Integrations

### LangChain (1-line drop-in)

```python
from adaptive_rag.integrations.langchain import AdaptiveRetriever

# Wrap your existing LangChain retriever
retriever = AdaptiveRetriever.from_corpus(docs, scope="my-app")

# Or use as a LangChain-compatible retriever directly
chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever.as_langchain_retriever(),
)
```

### LlamaIndex (1-line drop-in)

```python
from adaptive_rag.integrations.llamaindex import AdaptiveQueryEngine

# Wrap your existing LlamaIndex index
engine = AdaptiveQueryEngine.from_index(index, scope="my-app")
response = engine.query("What is the refund policy?")
```

### Haystack 2.x Component

```python
from adaptive_rag.integrations.haystack import AdaptiveHaystackRetriever

retriever = AdaptiveHaystackRetriever.from_texts(texts, scope="my-app")
# Add directly to Haystack Pipeline or run standalone:
result = retriever.run(query="What is the refund policy?", top_k=3)
docs = result["documents"]
```

### Direct use (no framework needed)

```python
from adaptive_rag import HybridRetriever, HashingEmbedder

# BM25 + dense semantic search, CPU-only
retriever = HybridRetriever(embedder=HashingEmbedder())
retriever.add(chunks)
results = retriever.search("refund policy", top_k=5)
```

---

## 🏥 Domain Starter Packs

Pre-configured retrieval for specific industries — no tuning required.

```python
# Healthcare: HIPAA-safe, no cloud, optimized for medical language
from adaptive_rag.packs import HealthcareRAG
rag = HealthcareRAG()
rag.add_documents("./patient_docs/")
results = rag.search("hypertension treatment guidelines")

# Legal: clause-aware chunking, precedent search
from adaptive_rag.packs import LegalRAG

# Education: multilingual, concept-aware
from adaptive_rag.packs import EducationRAG

# Code documentation: identifier-aware, semantic search
from adaptive_rag.packs import CodeDocsRAG
```

---

## 📊 Benchmark Results

Tested on the [Cranfield collection](https://ir.dcs.gla.ac.uk/resources/test_collections/cran/) — a standard IR benchmark with 1400 abstracts and 225 queries.

| Retriever | Recall@5 | MRR | Latency (p95) | API Cost |
|---|---|---|---|---|
| BM25 (baseline) | 0.72 | 0.65 | 8ms | \$0 |
| BM25 + Cache | 0.72 | 0.65 | **0.1ms** (cached) | \$0 |
| HybridRetriever | **0.84** | **0.79** | 12ms | \$0 |
| Typical cloud RAG | 0.81 | 0.76 | 120ms | \$200-800/mo |

> All benchmarks run on commodity CPU hardware. Reproduce with `adaptive-rag-evaluate --dataset benchmarks/cranfield.json`.

---

## 🛠️ Advanced Features

<details>
<summary><b>Hybrid Retrieval (BM25 + Dense)</b></summary>

```python
from adaptive_rag import HybridRetriever, HashingEmbedder

# Combine lexical and semantic search — no GPU required
retriever = HybridRetriever(
    embedder=HashingEmbedder(),
    anchor_boost=1.5,   # boost exact identifier matches
)
retriever.add(chunks)
results = retriever.search("machine learning optimization", top_k=10)
```
</details>

<details>
<summary><b>Persistent Disk Index (MMap)</b></summary>

```python
from adaptive_rag import MMapBM25Retriever

# Build once, read many times — survives restarts
MMapBM25Retriever.build("./my_index/", chunks)

# Open as context manager for automatic cleanup
with MMapBM25Retriever.open("./my_index/") as retriever:
    results = retriever.search("refund policy", top_k=5)
```
</details>

<details>
<summary><b>Metadata Filtering</b></summary>

```python
from adaptive_rag import BM25Retriever, MetadataFilteredRetriever, MetadataFilter

backend = BM25Retriever()
# Filter by department, date, user permissions, etc.
retriever = MetadataFilteredRetriever(
    backend,
    MetadataFilter(must={"department": "legal", "year": 2024})
)
```
</details>

<details>
<summary><b>Async / Concurrent Indexing</b></summary>

```python
from adaptive_rag import AsyncSegmentCoordinator

async with AsyncSegmentCoordinator("./index/") as coord:
    await coord.add_async(new_chunks)
    results = await coord.search_async("query", top_k=5)
```
</details>

---

## 📦 Installation

```bash
# Core (no dependencies)
pip install adaptive-rag

# With PDF support
pip install adaptive-rag[pdf]

# With image/OCR support (requires Tesseract)
pip install adaptive-rag[images]

# With NumPy SIMD acceleration
pip install adaptive-rag[numpy]

# Everything
pip install adaptive-rag[pdf,images,numpy]
```

---

## 🗺️ Roadmap

- [x] BM25 retrieval (CPU-only)
- [x] Semantic caching layer
- [x] Hybrid BM25 + dense retrieval
- [x] MMap disk-backed persistent index
- [x] Built-in evaluation CLI & RAG Doctor (`adaptive-rag doctor`)
- [x] Metadata filtering
- [x] Async indexing
- [x] LangChain adapter (`adaptive_rag.integrations.langchain`)
- [x] LlamaIndex adapter (`adaptive_rag.integrations.llamaindex`)
- [x] Haystack 2.x adapter (`adaptive_rag.integrations.haystack`)
- [x] Domain packs: Healthcare, Legal, Education, Code Docs, Finance
- [x] Document Loaders: PDF, OCR images, HTML, Word DOCX
- [x] Public leaderboard & Interactive Cost Calculator (`website/` and GitHub Pages)
- [x] TypeScript/JavaScript client bindings (`client-js/` package)
- [ ] Multilingual benchmark suite

---

## 🤝 Contributing

We love contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Good first issues:**
- Add a new file loader (DOCX, HTML, EPUB)
- Add a new benchmark dataset
- Improve evaluation metrics
- Write integration tutorials
- Translate README to another language

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Aravindraj93/adaptive-rag)

---

## 💬 Community

- 💬 [GitHub Discussions](https://github.com/Aravindraj93/adaptive-rag/discussions) — Ask questions, share results
- 🐛 [Issues](https://github.com/Aravindraj93/adaptive-rag/issues) — Bug reports and feature requests
- 📖 [API Reference](API.md) — Full API documentation
- 📊 [Benchmarks](benchmarks/) — Benchmark datasets and results

---

## 📝 License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

<div align="center">

**If adaptive-rag saves you money or time, please ⭐ star the repo — it helps others find it!**

Made with ❤️ for developers who believe great tools should be free and accessible to everyone.

</div>
