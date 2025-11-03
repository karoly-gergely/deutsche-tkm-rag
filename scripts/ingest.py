#!/usr/bin/env python3
"""Script to ingest documents into the vector store."""
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.vectorstores import Chroma

from config import settings
from core.chunking import MetadataAwareChunker
from core.embeddings import get_embeddings
from data.loader import DocumentLoader
from monitoring.logging import setup_logging

logger = setup_logging()


def main():
    """Main ingestion function."""
    parser = argparse.ArgumentParser(description="Ingest documents into vector store")
    parser.add_argument(
        "--data-folder",
        type=str,
        help="Path to data folder (overrides config)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Chunk size in characters (overrides config)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        help="Chunk overlap in characters (overrides config)",
    )
    args = parser.parse_args()

    # Determine data folder
    data_folder = args.data_folder or settings.DATA_FOLDER
    chunk_size = args.chunk_size or settings.CHUNK_SIZE
    chunk_overlap = args.chunk_overlap or settings.CHUNK_OVERLAP
    chroma_dir = settings.CHROMA_DIR

    try:
        # Create ChromaDB directory if missing
        os.makedirs(chroma_dir, exist_ok=True)
        logger.info(f"Using ChromaDB directory: {chroma_dir}")

        # Load documents
        logger.info(f"Loading documents from: {data_folder}")
        loader = DocumentLoader(data_folder=data_folder)
        documents = loader.load_all_documents()

        if not documents:
            print(f"⚠ No documents found in {data_folder}")
            logger.warning(f"No documents found in {data_folder}")
            sys.exit(1)

        logger.info(f"Loaded {len(documents)} document(s)")

        # Chunk documents with metadata-aware chunker
        logger.info(f"Chunking documents (size: {chunk_size}, overlap: {chunk_overlap})")
        chunker = MetadataAwareChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        all_chunks = []
        for doc in documents:
            # Extract source and doc_id from document metadata
            source = doc.metadata.get("source", "Deutsche Telekom")
            doc_id = doc.metadata.get("publication_id", doc.metadata.get("file_name", "unknown"))

            # Extract additional metadata to pass through
            extra_metadata = {
                k: v
                for k, v in doc.metadata.items()
                if k not in ["source", "publication_id", "file_name"]
            }

            # Chunk with metadata
            chunks = chunker.chunk_with_metadata(
                text=doc.page_content,
                source=source,
                doc_id=doc_id,
                **extra_metadata,
            )
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunk(s)")

        # Get embeddings
        embeddings = get_embeddings()

        # Build or load Chroma vectordb
        logger.info(f"Building ChromaDB vector store at {chroma_dir}")
        vectordb = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=chroma_dir,
        )

        # Persist
        vectordb.persist()
        logger.info("Vector store persisted successfully")

        # Print counts
        print(f"\n✓ Ingestion completed successfully!")
        print(f"  Documents: {len(documents)}")
        print(f"  Chunks: {len(all_chunks)}")
        print(f"  Vector store: {chroma_dir}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}", exc_info=True)
        print(f"✗ Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

