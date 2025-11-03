"""Data loading and processing package."""
from .loader import DocumentLoader
from .metadata import DocumentMetadata, MetadataExtractor, extract_metadata

__all__ = [
    "DocumentLoader",
    "DocumentMetadata",
    "MetadataExtractor",
    "extract_metadata",
]

