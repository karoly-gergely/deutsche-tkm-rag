"""
Core RAG (Retrieval-Augmented Generation) functionality package.

This package provides the foundational components for document processing and retrieval:
- Document chunking strategies (TextChunker, MetadataAwareChunker)
- Embedding model management and generation (EmbeddingModel, get_embeddings)
- Retrieval engines with advanced features (RetrievalEngine, AdvancedRetriever, Reranker)
"""

from .chunking import ChunkingStrategy, MetadataAwareChunker, TextChunker
from .embeddings import EmbeddingModel, get_embeddings
from .retrieval import (
    AdvancedRetriever,
    Reranker,
    RetrievalEngine,
    RetrievalError,
)

__all__ = [
    "EmbeddingModel",
    "get_embeddings",
    "ChunkingStrategy",
    "MetadataAwareChunker",
    "TextChunker",
    "AdvancedRetriever",
    "RetrievalError",
    "RetrievalEngine",
    "Reranker",
]
