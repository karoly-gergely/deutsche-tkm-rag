"""Dependency injection for API components (singletons)."""
from typing import Optional

from config import settings
from core.embeddings import get_embeddings
from core.retrieval import AdvancedRetriever

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

from llm.model_manager import ModelManager
from llm.prompt_manager import PromptManager
from monitoring.logging import StructuredLogger

# Global singletons (lazy-loaded)
_tokenizer: Optional[object] = None
_model: Optional[object] = None
_retriever: Optional[AdvancedRetriever] = None
_prompt_manager: Optional[PromptManager] = None
_logger: Optional[StructuredLogger] = None
_vectordb: Optional[Chroma] = None


def get_tokenizer():
    """Lazy-load tokenizer."""
    global _tokenizer, _model
    if _tokenizer is None:
        model_manager = ModelManager()
        _tokenizer, _model = model_manager.load_model()
    return _tokenizer


def get_model():
    """Lazy-load model."""
    global _model
    if _model is None:
        get_tokenizer()  # Triggers loading of both
    return _model


def get_vectordb():
    """Lazy-load vector database."""
    global _vectordb
    if _vectordb is None:
        embeddings = get_embeddings()
        _vectordb = Chroma(
            persist_directory=settings.CHROMA_DIR, embedding_function=embeddings
        )
    return _vectordb


def get_retriever():
    """Lazy-load retriever."""
    global _retriever
    if _retriever is None:
        vectordb = get_vectordb()
        _retriever = AdvancedRetriever(
            vectordb=vectordb, reranker_model=settings.RERANKER_MODEL
        )
    return _retriever


def get_prompt_manager():
    """Lazy-load prompt manager."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def get_logger():
    """Lazy-load logger."""
    global _logger
    if _logger is None:
        _logger = StructuredLogger("rag_api", log_dir=settings.LOG_DIR)
    return _logger
