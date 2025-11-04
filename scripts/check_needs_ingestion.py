#!/usr/bin/env python3
"""Check if ChromaDB needs ingestion (exits with status 1 if needed, 0 if not)."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

from core.embeddings import get_embeddings


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
            persist_directory=str(chroma_dir),
            embedding_function=embeddings
        )
        
        if hasattr(vectordb, "_collection") and vectordb._collection is not None:
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

