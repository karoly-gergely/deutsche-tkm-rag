#!/usr/bin/env python3
"""
Ingestion readiness check for deployment pipelines.
Verifies ChromaDB vector store existence and non-empty status,
exiting with appropriate status codes for CI/CD automation.
"""
import sys
from pathlib import Path

from config import settings
from core.embeddings import get_embeddings
from core.utils.imports import import_langchain_chroma

Chroma = import_langchain_chroma()


def needs_ingestion() -> bool:
    """Check if ChromaDB needs ingestion.

    Returns:
        True if ingestion is needed, False otherwise.
    """
    chroma_dir = Path(settings.CHROMA_DIR)

    # Check if directory doesn't exist or is empty
    if not chroma_dir.exists() or not any(chroma_dir.iterdir()):
        return True

    # Check if ChromaDB has vectors
    try:
        embeddings = get_embeddings()
        vectordb = Chroma(
            persist_directory=str(chroma_dir), embedding_function=embeddings
        )

        if (
            hasattr(vectordb, "_collection")
            and vectordb._collection is not None
        ):
            count = vectordb._collection.count()
            return count == 0

        return True  # No collection means we need ingestion
    except Exception:
        # If we can't check, assume we need ingestion
        return True


if __name__ == "__main__":
    if needs_ingestion():
        sys.exit(1)  # Needs ingestion
    else:
        sys.exit(0)  # Doesn't need ingestion
