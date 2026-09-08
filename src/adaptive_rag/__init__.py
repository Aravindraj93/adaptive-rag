"""CPU-first building blocks for adaptive retrieval."""

from .async_index import AsyncSegmentCoordinator
from .cache import CachedRetriever, CacheInfo
from .adaptive_fusion import AdaptiveFusionRetriever, FusionPolicy, FusionDecision
from .calibration import CalibrationReport, calibrate_confidence
from .chunking import TokenChunker
from .composition import ReciprocalRankFusionRetriever, SelectiveRerankingRetriever
from .embedders import HashingEmbedder
from .filtering import MetadataFilter, MetadataFilteredRetriever
from .hardware import HardwareProfile, profile_hardware
from .index_lifecycle import IndexUpdateReport, MMapIndexManager
from .models import Chunk, Document, Query, RetrievalStats, SearchResult
from .persistence import IndexFormatError
from .planner import AdaptiveRetriever, RetrievalDecision, RetrievalPlan
from .plugins import DocumentEnricher, PluginPipeline, QueryAnalyzer
from .retrievers.bm25 import BM25Retriever
from .retrievers.dense import DenseRetriever, Embedder
from .retrievers.mmap_bm25 import MMapBM25Retriever
from .retrievers.lsh_dense import LSHDenseRetriever
from .retrievers.mmap_dense import MMapDenseRetriever
from .retrievers.segmented_bm25 import SegmentedBM25Index, SegmentedBM25Retriever
from .routing import EscalatingRetriever, RouteDecision
from .telemetry import OperationMetric, TelemetryCollector
from .tokenization import NormalizedTokenizer

__all__ = [
    "CachedRetriever",
    "CacheInfo",
    "AdaptiveFusionRetriever",
    "FusionPolicy",
    "FusionDecision",
    "AdaptiveRetriever",
    "AsyncSegmentCoordinator",
    "BM25Retriever",
    "CalibrationReport",
    "Chunk",
    "DenseRetriever",
    "Document",
    "DocumentEnricher",
    "Embedder",
    "EscalatingRetriever",
    "HardwareProfile",
    "HashingEmbedder",
    "IndexFormatError",
    "IndexUpdateReport",
    "LSHDenseRetriever",
    "MMapBM25Retriever",
    "MMapDenseRetriever",
    "MMapIndexManager",
    "MetadataFilter",
    "MetadataFilteredRetriever",
    "NormalizedTokenizer",
    "OperationMetric",
    "PluginPipeline",
    "Query",
    "QueryAnalyzer",
    "ReciprocalRankFusionRetriever",
    "RetrievalDecision",
    "RetrievalPlan",
    "RetrievalStats",
    "RouteDecision",
    "SearchResult",
    "SegmentedBM25Index",
    "SegmentedBM25Retriever",
    "SelectiveRerankingRetriever",
    "TelemetryCollector",
    "TokenChunker",
    "calibrate_confidence",
    "profile_hardware",
]

__version__ = "0.11.0"
