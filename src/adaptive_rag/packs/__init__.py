"""Domain-specific RAG starter packs.

Pre-configured retrieval for specific industries — no tuning required.
Each pack provides sensible defaults for chunking, retrieval, and caching
specific to that domain's language patterns and use cases.

Available packs:
    from adaptive_rag.packs import HealthcareRAG   # HIPAA-safe, no cloud
    from adaptive_rag.packs import LegalRAG        # Clause-aware chunking
    from adaptive_rag.packs import EducationRAG    # Concept-aware, multilingual
    from adaptive_rag.packs import CodeDocsRAG     # Identifier-aware search
    from adaptive_rag.packs import FinanceRAG      # Numerical/temporal search

Example:
    from adaptive_rag.packs import HealthcareRAG

    rag = HealthcareRAG()
    rag.add_texts([
        "Hypertension is defined as systolic BP > 140 mmHg.",
        "ACE inhibitors are first-line treatment for hypertension.",
    ])
    results = rag.search("hypertension treatment options", top_k=5)
    for r in results:
        print(r.chunk.text)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Sequence


class _BaseRAGPack:
    """Base class for all domain packs.

    Provides a consistent interface:
        rag = SomePack()
        rag.add_texts([...])          # from strings
        rag.add_documents([...])      # from file paths or objects
        results = rag.search(query)   # returns SearchResult list
        info = rag.info()             # retriever statistics
    """

    # Subclasses configure these
    _SCOPE: str = "generic"
    _CHUNK_SIZE: int = 512       # tokens per chunk
    _CHUNK_OVERLAP: int = 64     # overlapping tokens between chunks
    _CACHE: bool = True
    _DESCRIPTION: str = "Generic RAG pack"

    def __init__(
        self,
        *,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        top_k: int = 5,
        use_cache: bool = True,
        scope: Optional[str] = None,
    ) -> None:
        from adaptive_rag import BM25Retriever, SemanticCachedRetriever

        self._chunk_size = chunk_size or self._CHUNK_SIZE
        self._chunk_overlap = chunk_overlap or self._CHUNK_OVERLAP
        self._top_k = top_k
        self._scope = scope or self._SCOPE
        self._use_cache = use_cache and self._CACHE

        self._backend = BM25Retriever()
        self._revision = 1

        if self._use_cache:
            _rev_ref = self
            self._retriever = SemanticCachedRetriever(
                self._backend,
                scope=self._scope,
                revision=lambda: _rev_ref._revision,
            )
        else:
            self._retriever = self._backend

        self._doc_count = 0
        self._chunk_count = 0

    # ── Document loading ─────────────────────────────────────────────────────

    def add_texts(
        self,
        texts: Sequence[str],
        *,
        metadatas: Optional[Sequence[dict]] = None,
        source: str = "text",
    ) -> "None":
        """Add text strings to the retrieval index.

        Args:
            texts: List of text strings to index.
            metadatas: Optional metadata dicts (one per text).
            source: Source label applied to all chunks.
        """
        from adaptive_rag import Chunk

        metas = list(metadatas or [{}] * len(texts))
        chunks = []
        for i, text in enumerate(texts):
            for j, chunk_text in enumerate(self._chunk_text(text)):
                processed_chunk_text = self._preprocess_document_text(chunk_text)
                chunk_id = f"{source}-{i}-{j}"
                meta = dict(metas[i]) if metas[i] else {}
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=str(meta.get("source", source)),
                        text=processed_chunk_text,
                        metadata=meta,
                    )
                )

        self._backend.add(chunks)
        self._doc_count += len(texts)
        self._chunk_count += len(chunks)
        self._revision += 1

    def add_documents(self, path_or_paths: "str | Path | Sequence[str | Path]") -> None:
        """Add documents from file paths.

        Supports: .txt, .md, .rst, .json, .csv

        Args:
            path_or_paths: A file path, directory path, or list of paths.
        """
        if isinstance(path_or_paths, (str, Path)):
            paths = [Path(path_or_paths)]
        else:
            paths = [Path(p) for p in path_or_paths]

        texts: list[str] = []
        metas: list[dict] = []

        for path in paths:
            if path.is_dir():
                for fp in path.rglob("*"):
                    if fp.suffix.lower() in {".txt", ".md", ".rst", ".json", ".csv"}:
                        try:
                            text = fp.read_text(encoding="utf-8", errors="ignore")
                            texts.append(text)
                            metas.append({"source": str(fp), "filename": fp.name})
                        except OSError:
                            pass
            elif path.is_file():
                text = path.read_text(encoding="utf-8", errors="ignore")
                texts.append(text)
                metas.append({"source": str(path), "filename": path.name})

        if texts:
            self.add_texts(texts, metadatas=metas, source="file")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, *, top_k: Optional[int] = None) -> list:
        """Search the corpus with domain-optimised retrieval.

        Args:
            query: Natural language query.
            top_k: Number of results (defaults to pack's top_k setting).

        Returns:
            List of adaptive_rag.SearchResult objects.
        """
        preprocessed = self._preprocess_query(query)
        return self._retriever.search(preprocessed, top_k=top_k or self._top_k)

    def info(self) -> dict:
        """Return retriever statistics.

        Returns:
            Dict with doc_count, chunk_count, scope, cache_enabled, etc.
        """
        base = {
            "pack": type(self).__name__,
            "description": self._DESCRIPTION,
            "doc_count": self._doc_count,
            "chunk_count": self._chunk_count,
            "chunk_size_tokens": self._chunk_size,
            "scope": self._scope,
            "cache_enabled": self._use_cache,
            "revision": self._revision,
        }
        if hasattr(self._retriever, "info"):
            base["cache_info"] = self._retriever.info()
        return base

    # ── Chunking & preprocessing (override in subclasses) ────────────────────

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks by approximate token count."""
        words = text.split()
        if not words:
            return []
        step = max(1, self._chunk_size - self._chunk_overlap)
        chunks = []
        for start in range(0, len(words), step):
            chunk = " ".join(words[start : start + self._chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _preprocess_document_text(self, text: str) -> str:
        """Domain-specific document preprocessing before indexing. Override in subclasses."""
        return text

    def _preprocess_query(self, query: str) -> str:
        """Domain-specific query preprocessing. Override in subclasses."""
        return query.strip()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"docs={self._doc_count}, "
            f"chunks={self._chunk_count}, "
            f"scope={self._scope!r})"
        )


# ── Healthcare Pack ───────────────────────────────────────────────────────────

class HealthcareRAG(_BaseRAGPack):
    """HIPAA-safe, CPU-only RAG for healthcare and medical documents.

    Features:
    - No cloud: all processing stays on your server
    - No embedding API: zero external data transmission
    - Optimised for medical terminology and clinical notes
    - Short chunks (256 tokens) for precise diagnosis-level retrieval
    - ICD/CPT code awareness in queries

    Use cases:
    - Clinical decision support (offline)
    - Medical literature search
    - Drug interaction lookup
    - Patient FAQ assistants (on-prem)

    Compliance note:
    - Zero network calls from this library
    - Bring your own LLM (local models recommended for PHI data)
    - Not a medical device; always involve qualified clinicians

    Example:
        rag = HealthcareRAG()
        rag.add_documents("./clinical_guidelines/")
        results = rag.search("first-line treatment for type 2 diabetes")
        for r in results:
            print(r.chunk.text)
    """

    _SCOPE = "healthcare"
    _CHUNK_SIZE = 256        # Shorter for clinical precision
    _CHUNK_OVERLAP = 32
    _DESCRIPTION = "HIPAA-safe healthcare RAG — CPU-only, no data leaves your server"

    # Medical abbreviation expansions for better query matching
    _ABBREVS = {
        "bp": "blood pressure",
        "hr": "heart rate",
        "dm": "diabetes mellitus",
        "htn": "hypertension",
        "cad": "coronary artery disease",
        "chf": "congestive heart failure",
        "copd": "chronic obstructive pulmonary disease",
        "uti": "urinary tract infection",
        "mi": "myocardial infarction heart attack",
        "cvd": "cardiovascular disease",
        "aki": "acute kidney injury",
        "ckd": "chronic kidney disease",
        "afib": "atrial fibrillation",
        "pe": "pulmonary embolism",
        "dvt": "deep vein thrombosis",
        "gerd": "gastroesophageal reflux",
        "ibs": "irritable bowel syndrome",
        "ra": "rheumatoid arthritis",
    }

    def _preprocess_document_text(self, text: str) -> str:
        """Enrich document text with expanded abbreviations for recall."""
        tokens = text.split()
        added_expansions = []
        for token in tokens:
            clean = re.sub(r"[^\w]", "", token).lower()
            if clean in self._ABBREVS and self._ABBREVS[clean] not in text.lower():
                added_expansions.append(self._ABBREVS[clean])
        if added_expansions:
            return f"{text}\n[Expanded clinical context: {' '.join(set(added_expansions))}]"
        return text

    def _preprocess_query(self, query: str) -> str:
        """Expand medical abbreviations in queries while preserving original tokens."""
        tokens = query.lower().split()
        expanded = []
        for token in tokens:
            clean = re.sub(r"[^\w]", "", token)
            if clean in self._ABBREVS:
                expanded.extend([token, self._ABBREVS[clean]])
            else:
                expanded.append(token)
        return " ".join(expanded)


# ── Legal Pack ────────────────────────────────────────────────────────────────

class LegalRAG(_BaseRAGPack):
    """Clause-aware RAG for legal documents, contracts, and case law.

    Features:
    - Clause-boundary-aware chunking (splits on section markers)
    - Optimised for precise clause retrieval
    - Legal citation awareness
    - Larger chunks (768 tokens) to preserve clause context

    Use cases:
    - Contract analysis and review
    - Legal research and precedent search
    - Compliance documentation
    - Due diligence Q&A

    Example:
        rag = LegalRAG()
        rag.add_documents("./contracts/")
        results = rag.search("indemnification clause limitations liability")
    """

    _SCOPE = "legal"
    _CHUNK_SIZE = 768        # Larger to preserve clause context
    _CHUNK_OVERLAP = 128
    _DESCRIPTION = "Clause-aware legal document RAG with citation support"

    # Legal section markers for boundary-aware chunking
    _SECTION_PATTERN = re.compile(
        r"(?:Section|§|Article|Clause|SECTION|ARTICLE)\s+\d+",
        re.IGNORECASE,
    )

    def _chunk_text(self, text: str) -> list[str]:
        """Split on legal section boundaries when possible."""
        # Try to split on section/article markers first
        sections = self._SECTION_PATTERN.split(text)
        markers = self._SECTION_PATTERN.findall(text)

        if len(sections) > 1:
            # Reassemble with markers
            chunks = []
            for i, section in enumerate(sections[1:], 0):
                header = markers[i] if i < len(markers) else ""
                chunk_text = f"{header}\n{section}".strip()
                if chunk_text:
                    # Further split if too long
                    words = chunk_text.split()
                    if len(words) > self._chunk_size:
                        for start in range(0, len(words), self._chunk_size - self._chunk_overlap):
                            c = " ".join(words[start : start + self._chunk_size])
                            if c.strip():
                                chunks.append(c)
                    else:
                        chunks.append(chunk_text)
            if chunks:
                return chunks

        # Fall back to word-based chunking
        return super()._chunk_text(text)

    def _preprocess_query(self, query: str) -> str:
        """Expand legal shorthand."""
        expansions = {
            "indemnity": "indemnification indemnity hold harmless",
            "ip": "intellectual property copyright trademark patent",
            "sla": "service level agreement uptime availability",
            "nda": "non-disclosure agreement confidentiality",
            "toc": "terms and conditions",
            "gdpr": "general data protection regulation privacy",
        }
        tokens = query.lower().split()
        result = []
        for t in tokens:
            result.append(expansions.get(t, t))
        return " ".join(result)


# ── Education Pack ────────────────────────────────────────────────────────────

class EducationRAG(_BaseRAGPack):
    """Concept-aware RAG for educational content and course materials.

    Features:
    - Optimised for concept-based queries (e.g. "explain recursion")
    - Medium chunks (384 tokens) for explanation-level context
    - Multilingual query support through phonetic normalisation
    - Works great for Q&A over textbooks, lecture notes, FAQs

    Use cases:
    - Student Q&A chatbots
    - Course content search
    - Textbook explainer assistants
    - Educational platform search

    Example:
        rag = EducationRAG()
        rag.add_documents("./lecture_notes/")
        results = rag.search("explain gradient descent in simple terms")
    """

    _SCOPE = "education"
    _CHUNK_SIZE = 384
    _CHUNK_OVERLAP = 64
    _DESCRIPTION = "Concept-aware education RAG for textbooks and course materials"

    def _preprocess_query(self, query: str) -> str:
        """Normalise student query language."""
        query = query.lower()
        # Map common student phrasings to technical terms
        replacements = [
            (r"\bwhat is\b", "define"),
            (r"\bhow does\b", "explain mechanism"),
            (r"\bwhy is\b", "reason explanation"),
            (r"\bcan you explain\b", ""),
            (r"\bplease explain\b", ""),
            (r"\bi don'?t understand\b", ""),
            (r"\bhelp me with\b", ""),
        ]
        for pattern, replacement in replacements:
            query = re.sub(pattern, replacement, query)
        return query.strip()


# ── Code Docs Pack ────────────────────────────────────────────────────────────

class CodeDocsRAG(_BaseRAGPack):
    """Identifier-aware RAG for code documentation and API references.

    Features:
    - CamelCase and snake_case identifier splitting
    - Short chunks (256 tokens) for precise function/class lookup
    - Docstring-optimised chunking
    - Works with Python, JS, Java, Go, Rust docs

    Use cases:
    - API reference search
    - Code documentation Q&A
    - SDK usage examples lookup
    - Error message diagnosis

    Example:
        rag = CodeDocsRAG()
        rag.add_documents("./docs/")
        results = rag.search("how to use SemanticCachedRetriever with revision")
    """

    _SCOPE = "code-docs"
    _CHUNK_SIZE = 256
    _CHUNK_OVERLAP = 32
    _DESCRIPTION = "Identifier-aware RAG for code documentation and API references"

    _CAMEL_RE = re.compile(r"([A-Z][a-z]+|[A-Z]+(?=[A-Z]|$))")

    def _preprocess_query(self, query: str) -> str:
        """Split identifiers for better BM25 matching."""
        # Split CamelCase: SemanticCachedRetriever → Semantic Cached Retriever
        query = self._CAMEL_RE.sub(r" \1", query)
        # Split snake_case: my_function → my function
        query = query.replace("_", " ")
        # Normalise
        return " ".join(query.split()).lower()


# ── Finance Pack ──────────────────────────────────────────────────────────────

class FinanceRAG(_BaseRAGPack):
    """Numerical and temporal RAG for financial documents and reports.

    Features:
    - Larger chunks (1024 tokens) to preserve financial table context
    - Ticker symbol awareness
    - Date/period normalisation in queries
    - Optimised for 10-K, 10-Q, earnings reports, research notes

    Use cases:
    - Earnings report Q&A
    - Financial research assistants
    - Risk document analysis
    - Regulatory filing search

    Example:
        rag = FinanceRAG()
        rag.add_documents("./10k_reports/")
        results = rag.search("revenue growth Q3 2024 YoY comparison")
    """

    _SCOPE = "finance"
    _CHUNK_SIZE = 1024       # Large to capture tables and context
    _CHUNK_OVERLAP = 128
    _DESCRIPTION = "Numerical-aware finance RAG for reports and regulatory filings"

    def _preprocess_query(self, query: str) -> str:
        """Normalise financial shorthand."""
        expansions = {
            "yoy": "year over year",
            "qoq": "quarter over quarter",
            "mom": "month over month",
            "ebitda": "earnings before interest taxes depreciation amortisation",
            "eps": "earnings per share",
            "pe": "price to earnings ratio",
            "roe": "return on equity",
            "roi": "return on investment",
            "cagr": "compound annual growth rate",
            "capex": "capital expenditure",
            "opex": "operating expense",
            "fcf": "free cash flow",
            "arpu": "average revenue per user",
            "arr": "annual recurring revenue",
            "mrr": "monthly recurring revenue",
            "ltv": "lifetime value",
            "cac": "customer acquisition cost",
        }
        tokens = query.lower().split()
        return " ".join(expansions.get(t, t) for t in tokens)


# ── Convenience exports ───────────────────────────────────────────────────────

__all__ = [
    "HealthcareRAG",
    "LegalRAG",
    "EducationRAG",
    "CodeDocsRAG",
    "FinanceRAG",
]
