"""Retrieval and reranking functionality."""
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.docstore.document import Document
    except ImportError:
        from langchain.schema import Document

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

from config import settings
from core.embeddings import EmbeddingModel

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

# Try both langchain 0.1.x and 0.2+ import paths
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
    except ImportError:
        raise ImportError(
            "Could not import HuggingFaceEmbeddings. "
            "Please install langchain or langchain-community."
        )


class RetrievalError(Exception):
    """Custom exception for retrieval operations."""

    pass


class Reranker:
    """Reranker using cross-encoder models."""

    def __init__(self, model_name: str | None = None):
        """Initialize reranker.

        Args:
            model_name: Name of the reranker model. Defaults to settings value.
        """
        self.model_name = model_name or settings.RERANKER_MODEL
        self._model = None

    @property
    def model(self):
        """Lazy load the reranker model."""
        if CrossEncoder is None:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Install with: pip install sentence-transformers"
            )
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, documents: List[str], top_k: int | None = None
    ) -> List[tuple[str, float]]:
        """Rerank documents based on relevance to query.

        Args:
            query: Search query.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. Defaults to settings value.

        Returns:
            List of (document, score) tuples sorted by relevance.
        """
        if not documents:
            return []

        top_k = top_k or settings.RERANK_TOP_K

        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)

        scored_docs = list(zip(documents, scores.tolist()))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:top_k]


class RetrievalEngine:
    """Vector store-based retrieval engine."""

    def __init__(
        self,
        persist_directory: str | None = None,
        embedding_model: EmbeddingModel | None = None,
    ):
        """Initialize retrieval engine.

        Args:
            persist_directory: Directory to persist ChromaDB data.
            embedding_model: Embedding model instance.
        """
        self.persist_directory = persist_directory or settings.CHROMA_DIR
        self.embedding_model = embedding_model or EmbeddingModel()

        # Initialize LangChain embeddings wrapper
        self.langchain_embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )
        self._vectordb: Optional[Chroma] = None

    @property
    def vectordb(self) -> Chroma:
        """Get or create the vector database."""
        if self._vectordb is None:
            self._vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.langchain_embeddings,
            )
        return self._vectordb

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of Document objects to add.
        """
        self.vectordb.add_documents(documents)
        self.vectordb.persist()

    def similarity_search(
        self, query: str, k: int | None = None, rerank: bool = False
    ) -> List[Document]:
        """Perform similarity search.

        Args:
            query: Search query.
            k: Number of results to return. Defaults to settings value.
            rerank: Whether to apply reranking.

        Returns:
            List of relevant documents.
        """
        k = k or settings.TOP_K

        results = self.vectordb.similarity_search(query, k=k * 2 if rerank else k)

        if rerank:
            reranker = Reranker()
            doc_texts = [doc.page_content for doc in results]
            reranked = reranker.rerank(query, doc_texts, top_k=k)

            # Map back to documents
            text_to_doc = {doc.page_content: doc for doc in results}
            results = [text_to_doc[text] for text, _ in reranked if text in text_to_doc]

        return results[:k]


class AdvancedRetriever:
    """Advanced retriever with reranking and safe fallback support."""

    def __init__(self, vectordb: Chroma, reranker_model: Optional[str] = None):
        """Initialize advanced retriever.

        Args:
            vectordb: Chroma vector database instance.
            reranker_model: Optional reranker model name. If None, reranking is disabled.
        """
        self.vectordb = vectordb
        self.reranker_model = reranker_model
        self._reranker = None

    @property
    def reranker(self):
        """Lazy load the reranker model.

        Returns:
            CrossEncoder instance if reranker_model is set.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        if self.reranker_model is None:
            return None

        if self._reranker is None:
            if CrossEncoder is None:
                raise ImportError(
                    "sentence-transformers is required for reranking. "
                    "Install with: pip install sentence-transformers"
                )
            self._reranker = CrossEncoder(
                self.reranker_model, max_length=512
            )
        return self._reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        rerank_top_k: Optional[int] = None,
        filters: Optional[Dict] = None,
    ) -> List[Document]:
        """Retrieve documents with optional reranking.

        Args:
            query: Search query.
            top_k: Number of final results to return.
            rerank_top_k: Number of candidates to retrieve before reranking.
                If None, uses top_k. Only used if reranker is enabled.
            filters: Optional metadata filters for similarity search.

        Returns:
            List of retrieved documents, optionally reranked.
        """
        # Determine initial retrieval count: rerank_top_k or top_k
        if self.reranker is not None:
            k = rerank_top_k if rerank_top_k is not None else top_k
        else:
            k = top_k

        # Perform similarity search
        results = self.vectordb.similarity_search(
            query, k=k, filter=filters
        )

        # Apply reranking if reranker is available and we have more results than needed
        if self.reranker is not None and len(results) > top_k:
            # Extract document texts
            doc_texts = [doc.page_content for doc in results]

            # Score query-document pairs: [query, doc.page_content]
            pairs = [[query, doc_text] for doc_text in doc_texts]
            scores = self.reranker.predict(pairs)

            # Sort by score (descending) using numpy and return top_k
            indices = np.argsort(scores)[::-1]  # Descending order
            sorted_results = [results[i] for i in indices[:top_k]]

            return sorted_results

        return results[:top_k]

    def retrieve_with_scores(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Retrieve documents with similarity scores.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            List of (Document, score) tuples sorted by score (highest first).

        Raises:
            RetrievalError: If retrieval fails.
        """
        try:
            results_with_scores = self.vectordb.similarity_search_with_score(
                query, k=top_k
            )

            # Convert to list of tuples and ensure scores are floats
            return [
                (doc, float(score))
                for doc, score in results_with_scores
            ]
        except Exception as e:
            raise RetrievalError(f"Failed to retrieve with scores: {e}") from e

    def retrieve_safe(
        self, query: str, top_k: int = 5, fallback_k: int = 3
    ) -> List[Document]:
        """Retrieve documents with safe fallback on errors.

        Args:
            query: Search query.
            top_k: Primary number of results to return.
            fallback_k: Number of results to return if primary retrieval fails.

        Returns:
            List of retrieved documents. Returns fewer results if fallback is used.
        """
        try:
            # Attempt primary retrieval
            results = self.vectordb.similarity_search(query, k=top_k)
            return results
        except Exception as e:
            # Fallback to smaller retrieval
            try:
                fallback_results = self.vectordb.similarity_search(
                    query, k=fallback_k
                )
                return fallback_results
            except Exception as fallback_error:
                raise RetrievalError(
                    f"Both primary and fallback retrieval failed. "
                    f"Primary error: {e}, Fallback error: {fallback_error}"
                ) from fallback_error

