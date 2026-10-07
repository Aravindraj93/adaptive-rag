# Contributing to adaptive-rag

Thank you for considering contributing! adaptive-rag exists to make production-quality RAG retrieval accessible to everyone — no GPU, no cloud, no API bills. Your contributions help make that real.

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/Aravindraj93/adaptive-rag
cd adaptive-rag

# Install in editable mode with dev dependencies
pip install -e ".[numpy]"
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# Run the RAG Doctor smoke test
mkdir -p /tmp/test-corpus
echo "Test document for health check." > /tmp/test-corpus/test.txt
adaptive-rag doctor --corpus /tmp/test-corpus
```

## 🎯 Good First Issues

New to the project? These are great places to start:

| Label | Description | Difficulty |
|---|---|---|
| `good first issue` | Small, self-contained tasks | ⭐ Easy |
| `docs` | Improve or add documentation | ⭐ Easy |
| `integration` | Add framework adapters | ⭐⭐ Medium |
| `benchmark` | Add benchmark datasets | ⭐⭐ Medium |
| `loader` | Add new file format loaders | ⭐ Easy |
| `pack` | Add a new domain pack | ⭐⭐ Medium |

**Specifically looking for:**
- [ ] DOCX file loader (`src/adaptive_rag/loaders.py`)
- [ ] HTML file loader
- [ ] EPUB file loader
- [ ] Haystack framework adapter (`src/adaptive_rag/integrations/haystack.py`)
- [ ] DSPy framework adapter
- [ ] SciFact benchmark dataset integration
- [ ] MS MARCO benchmark integration
- [ ] BEIR benchmark integration
- [ ] Translate README to: Spanish, French, Portuguese, Chinese, Japanese, Hindi, Arabic
- [ ] Add multilingual BM25 tokenization (via `snowballstemmer`)

## 📁 Project Structure

```
adaptive-rag/
├── src/adaptive_rag/
│   ├── __init__.py          # Public API surface
│   ├── doctor.py            # RAG Doctor CLI tool
│   ├── benchmark.py         # Evaluation harness
│   ├── models.py            # Chunk, Query, SearchResult
│   ├── retrievers/          # BM25, Dense, MMap retrievers
│   ├── integrations/        # Framework adapters
│   │   ├── langchain.py     # LangChain adapter
│   │   └── llamaindex.py    # LlamaIndex adapter
│   ├── packs/               # Domain-specific starter packs
│   │   └── __init__.py      # HealthcareRAG, LegalRAG, etc.
│   ├── cache.py             # CachedRetriever
│   ├── semantic_cache.py    # SemanticCachedRetriever
│   ├── hybrid.py            # HybridRetriever (BM25 + dense)
│   ├── loaders.py           # File loaders (PDF, text, JSON, ...)
│   └── chunking.py          # TokenChunker
├── tests/                   # Test suite
├── benchmarks/              # Benchmark datasets (JSON)
├── examples/                # Usage examples
├── docs/                    # Documentation
└── .github/workflows/       # CI/CD
```

## ✅ How to Add a New Integration

1. Create `src/adaptive_rag/integrations/<framework>.py`
2. Implement a class that converts between adaptive-rag objects and framework objects
3. Add a factory classmethod for easy instantiation
4. Add tests in `tests/integrations/test_<framework>.py`
5. Add an example in `examples/integrations/<framework>_example.py`
6. Update `README.md` with a usage snippet

**Template:**
```python
# src/adaptive_rag/integrations/myframework.py

class AdaptiveMyFrameworkRetriever:
    """Adapter for MyFramework."""

    @classmethod
    def from_corpus(cls, documents, *, scope="my-app", top_k=5):
        from adaptive_rag import BM25Retriever, Chunk, SemanticCachedRetriever
        chunks = [Chunk(str(i), "docs", doc.text) for i, doc in enumerate(documents)]
        backend = BM25Retriever()
        backend.add(chunks)
        retriever = SemanticCachedRetriever(backend, scope=scope, revision=lambda: 1)
        return cls(retriever, top_k=top_k)

    def search(self, query, top_k=None):
        return self._retriever.search(query, top_k=top_k or self._top_k)
```

## ✅ How to Add a New Domain Pack

1. Add your pack class to `src/adaptive_rag/packs/__init__.py`
2. Extend `_BaseRAGPack`
3. Set `_SCOPE`, `_CHUNK_SIZE`, `_CHUNK_OVERLAP`, `_DESCRIPTION`
4. Override `_preprocess_query()` with domain-specific normalization
5. Optionally override `_chunk_text()` for domain-specific chunking
6. Add to `__all__` and update `README.md`

## ✅ How to Add a New Benchmark Dataset

1. Create `benchmarks/<name>.json` with this format:
```json
[
  {
    "query": "What causes hypertension?",
    "relevant_ids": ["doc-1", "doc-5"],
    "corpus": [
      {"id": "doc-1", "text": "Hypertension is caused by..."},
      {"id": "doc-5", "text": "Risk factors include..."}
    ]
  }
]
```
2. Test it: `adaptive-rag-evaluate --dataset benchmarks/<name>.json`
3. Add a note to `benchmarks/README.md`

## 🧪 Writing Tests

```python
# tests/test_my_feature.py
from adaptive_rag import BM25Retriever, Chunk

def test_basic_retrieval():
    retriever = BM25Retriever()
    retriever.add([
        Chunk("c1", "docs", "The sky is blue."),
        Chunk("c2", "docs", "The grass is green."),
    ])
    results = retriever.search("sky color", top_k=1)
    assert len(results) == 1
    assert results[0].chunk.id == "c1"
```

Run tests:
```bash
pytest tests/ -v                    # All tests
pytest tests/test_doctor.py -v      # Specific file
pytest -k "cache" -v                # Tests matching pattern
```

## 📝 Code Style

- **No mandatory new dependencies** — if you add a dep, it MUST be optional (extra)
- Follow existing code style (no linter config; just match the surrounding code)
- Add type hints to all public functions
- Write docstrings for all public classes and methods
- Keep CPU-first: never add a mandatory GPU operation

## 🔄 Pull Request Process

1. Fork and create a branch: `git checkout -b feature/my-feature`
2. Make changes and add tests
3. Run `pytest tests/ -v` and ensure all tests pass
4. Update `CHANGELOG.md` under `[Unreleased]`
5. Open a PR with a clear description of what and why

## 💬 Need Help?

- [GitHub Discussions](https://github.com/Aravindraj93/adaptive-rag/discussions) — Questions, ideas, show-and-tell
- [Issues](https://github.com/Aravindraj93/adaptive-rag/issues) — Bug reports

## 📜 License

By contributing, you agree your contributions will be licensed under the Apache 2.0 License.

---

**Thank you for making RAG accessible to everyone! 🙏**
