"""Append, delete, search, and compact an immutable segmented index."""

from pathlib import Path
from tempfile import TemporaryDirectory

from adaptive_rag import Chunk, SegmentedBM25Index, SegmentedBM25Retriever


with TemporaryDirectory() as directory:
    path = Path(directory) / "search-index"
    manager = SegmentedBM25Index.create(
        path,
        [Chunk("first", "guide", "Initial searchable guidance")],
    )
    manager.append([Chunk("second", "policy", "Updated return policy")])
    manager.delete(["first"])

    with SegmentedBM25Retriever(path) as retriever:
        print(retriever.search("return policy")[0].chunk.id)

    manager.compact()
