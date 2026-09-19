"""CPU-first building blocks for adaptive retrieval."""

from .async_index import AsyncSegmentCoordinator
from .cache import CachedRetriever, CacheInfo
from .adaptive_fusion import AdaptiveFusionRetriever, FusionPolicy, FusionDecision
from .calibration import CalibrationReport, calibrate_confidence
from .chunking import TokenChunker
from .composition import (
    ReciprocalRankFusionRetriever,
    RelationalExpansionRetriever,
    ScoreWeightedFusionRetriever,
    SelectiveRerankingRetriever,
)
from .embedders import HashingEmbedder
from .filtering import MetadataFilter, MetadataFilteredRetriever
from .hardware import HardwareProfile, profile_hardware
from .hybrid import HybridRetriever
from .index_lifecycle import IndexUpdateReport, MMapIndexManager
from .loaders import (
    BaseLoader,
    DirectoryLoader,
    ImageLoader,
    JSONLoader,
    PDFLoader,
    TextLoader,
    load_file,
)
from .models import Chunk, Document, Query, RetrievalStats, SearchResult
from .persistence import IndexFormatError
from .planner import AdaptiveRetriever, RetrievalDecision, RetrievalPlan
from .plugins import (
    DocumentEnricher,
    FastPathIDLookupRetriever,
    MetadataScoreModifier,
    PluginPipeline,
    QueryAnalyzer,
)
from .retrievers.bm25 import BM25Retriever
from .retrievers.dense import DenseRetriever, Embedder
from .retrievers.mmap_bm25 import MMapBM25Retriever
from .retrievers.lsh_dense import LSHDenseRetriever
from .retrievers.mmap_dense import MMapDenseRetriever
from .retrievers.segmented_bm25 import SegmentedBM25Index, SegmentedBM25Retriever
from .routing import EscalatingRetriever, RouteDecision
from .telemetry import OperationMetric, TelemetryCollector
from .tokenization import CharNGramTokenizer, NormalizedTokenizer

__all__ = [
    "BaseLoader",
    "CachedRetriever",
    "CacheInfo",
    "AdaptiveFusionRetriever",
    "FusionPolicy",
    "FusionDecision",
    "AdaptiveRetriever",
    "AsyncSegmentCoordinator",
    "BM25Retriever",
    "CalibrationReport",
    "CharNGramTokenizer",
    "Chunk",
    "DenseRetriever",
    "DirectoryLoader",
    "Document",
    "DocumentEnricher",
    "Embedder",
    "EscalatingRetriever",
    "FastPathIDLookupRetriever",
    "HardwareProfile",
    "HashingEmbedder",
    "HybridRetriever",
    "ImageLoader",
    "IndexFormatError",
    "IndexUpdateReport",
    "JSONLoader",
    "LSHDenseRetriever",
    "MMapBM25Retriever",
    "MMapDenseRetriever",
    "MMapIndexManager",
    "MetadataFilter",
    "MetadataFilteredRetriever",
    "MetadataScoreModifier",
    "NormalizedTokenizer",
    "OperationMetric",
    "PDFLoader",
    "PluginPipeline",
    "Query",
    "QueryAnalyzer",
    "ReciprocalRankFusionRetriever",
    "RelationalExpansionRetriever",
    "RetrievalDecision",
    "RetrievalPlan",
    "RetrievalStats",
    "RouteDecision",
    "ScoreWeightedFusionRetriever",
    "SearchResult",
    "SegmentedBM25Index",
    "SegmentedBM25Retriever",
    "SelectiveRerankingRetriever",
    "TelemetryCollector",
    "TextLoader",
    "TokenChunker",
    "calibrate_confidence",
    "load_file",
    "profile_hardware",
]

__version__ = "0.12.0"
