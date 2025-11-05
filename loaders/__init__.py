"""
Data loading and document processing package.

Provides utilities for loading documents from various sources and extracting metadata:
- Document loading from multiple formats (DocumentLoader)
- Metadata extraction and management (MetadataExtractor, DocumentMetadata, extract_metadata)
"""

from .loader import DocumentLoader
from .metadata import DocumentMetadata, MetadataExtractor, extract_metadata

__all__ = [
    "DocumentLoader",
    "DocumentMetadata",
    "MetadataExtractor",
    "extract_metadata",
]
