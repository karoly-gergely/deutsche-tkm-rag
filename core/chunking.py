"""
Text segmentation strategies for document processing.
Implements metadata-aware chunking with overlap preservation
for vector store indexing using LangChain document structures.
"""

import hashlib
from abc import ABC, abstractmethod
from typing import Any

from core.utils.imports import (
    import_langchain_document_class,
    import_langchain_recursive_character_text_splitter,
)

Document = import_langchain_document_class()
RecursiveCharacterTextSplitter = (
    import_langchain_recursive_character_text_splitter()
)


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split text into chunks.

        Args:
            text: Input text to chunk.

        Returns:
            List of text chunks.
        """
        pass


class TextChunker(ChunkingStrategy):
    """Simple word-based chunking with overlap."""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """Initialize chunker.

        Args:
            chunk_size: Target size of chunks in words.
            chunk_overlap: Number of words to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        """Split text into chunks with overlap.

        Args:
            text: Input text to chunk.

        Returns:
            List of text chunks.
        """
        words = text.split()
        chunks = []
        step = self.chunk_size - self.chunk_overlap

        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks if chunks else [text]


class MetadataAwareChunker:
    """Metadata-aware chunker using RecursiveCharacterTextSplitter."""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """Initialize metadata-aware chunker.

        Args:
            chunk_size: Target size of chunks in characters.
            chunk_overlap: Number of characters to overlap between chunks.
        """
        if RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "RecursiveCharacterTextSplitter not available. "
                "Please install langchain or langchain-community."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @staticmethod
    def _generate_chunk_id(
        doc_id: str, chunk_index: int, chunk_text: str
    ) -> str:
        """Generate stable chunk ID from doc_id, index, and first 50 chars.

        Args:
            doc_id: Document identifier.
            chunk_index: Index of the chunk.
            chunk_text: Text content of the chunk.

        Returns:
            First 16 hex characters of SHA256 hash.
        """
        first_50_chars = chunk_text[:50] if chunk_text else ""
        hash_input = f"{doc_id}_{chunk_index}_{first_50_chars}"
        hash_value = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
        return hash_value[:16]

    def chunk_with_metadata(
        self, text: str, source: str, doc_id: str, **metadata: Any
    ) -> list[Document]:
        """Split text into chunks with consistent metadata.

        Args:
            text: Input text to chunk.
            source: Source identifier for the document.
            doc_id: Unique document identifier.
            **metadata: Additional metadata to include in each chunk.

        Returns:
            List of Document objects with consistent metadata.

        Note:
            Each chunk will have: source, doc_id, chunk_index, total_chunks,
            chunk_id, and any additional metadata passed via **metadata.
        """
        # Split text into chunks
        chunks = self.splitter.split_text(text)
        total_chunks = len(chunks)

        # Create Document objects with metadata
        documents = []
        for idx, chunk_text in enumerate(chunks):
            # Generate stable chunk ID
            chunk_id = self._generate_chunk_id(
                doc_id=doc_id, chunk_index=idx, chunk_text=chunk_text
            )

            # Build metadata dictionary
            chunk_metadata: dict[str, Any] = {
                "source": source,
                "doc_id": doc_id,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "chunk_id": chunk_id,
            }

            # Add any additional metadata
            chunk_metadata.update(metadata)

            # Create Document
            doc = Document(page_content=chunk_text, metadata=chunk_metadata)
            documents.append(doc)

        return documents
