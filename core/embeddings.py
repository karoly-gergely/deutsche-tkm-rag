"""Embedding model management."""
from typing import List

from sentence_transformers import SentenceTransformer

from config import settings

# Try both langchain 0.1.x and 0.2+ import paths
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None


def get_embeddings():
    """Get HuggingFaceEmbeddings instance for LangChain integration.

    Returns:
        HuggingFaceEmbeddings instance configured with settings.EMBEDDING_MODEL.

    Raises:
        ImportError: If HuggingFaceEmbeddings cannot be imported.
    """
    if HuggingFaceEmbeddings is None:
        raise ImportError(
            "HuggingFaceEmbeddings not available. "
            "Please install langchain or langchain-community."
        )
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding models."""

    def __init__(self, model_name: str | None = None):
        """Initialize embedding model.

        Args:
            model_name: Name of the model to use. Defaults to settings value.
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Encode texts into embeddings.

        Args:
            texts: List of text strings to encode.
            **kwargs: Additional arguments to pass to the model.

        Returns:
            List of embedding vectors.
        """
        return self.model.encode(texts, **kwargs).tolist()

    def encode_single(self, text: str, **kwargs) -> List[float]:
        """Encode a single text into an embedding.

        Args:
            text: Text string to encode.
            **kwargs: Additional arguments to pass to the model.

        Returns:
            Embedding vector.
        """
        return self.encode([text], **kwargs)[0]

