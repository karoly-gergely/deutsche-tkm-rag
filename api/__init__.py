"""
REST API package for the Deutsche Telekom RAG application.

Provides FastAPI endpoints for:
- Document ingestion and management
- Query processing and retrieval
- Health checks and diagnostics
"""

from .routes import app

__all__ = ["app"]
