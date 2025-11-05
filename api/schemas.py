"""
Pydantic models (schemas) for API request and response validation.
Defines the structure of the data exchanged between the API and clients.
"""

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Message model for conversation history."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str = Field(
        ..., min_length=1, max_length=1000, description="User query"
    )
    top_k: int | None = Field(
        None,
        ge=1,
        le=20,
        description="Number of documents to retrieve (clamped to 1-20)",
    )
    messages: list[Message] | None = Field(
        None,
        description="Optional conversation history for context-aware responses",
    )


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    answer: str
    sources: list[str]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    sizes: dict[str, int | None]


class SettingsResponse(BaseModel):
    """Settings response model."""

    model_id: str = Field(..., description="Language model ID")
    embedding_model: str = Field(..., description="Embedding model ID")
    device: str = Field(..., description="Device for model execution")
    chunk_size: int = Field(..., description="Document chunk size")
    chunk_overlap: int = Field(..., description="Document chunk overlap")
    top_k: int = Field(
        ..., description="Default number of documents to retrieve"
    )
    rerank_top_k: int | None = Field(
        None, description="Number of candidates for reranking"
    )
    max_context_tokens: int = Field(..., description="Maximum context tokens")
    max_new_tokens: int = Field(
        ..., description="Maximum new tokens to generate"
    )
    temperature: float = Field(..., description="Sampling temperature")
    top_p: float = Field(..., description="Nucleus sampling parameter")
    data_folder: str = Field(..., description="Data folder path")
    chroma_dir: str = Field(..., description="ChromaDB directory")
    log_level: str = Field(..., description="Logging level")
    enable_tracing: bool = Field(..., description="Whether tracing is enabled")
    log_dir: str = Field(..., description="Log directory")
    reranker_model: str | None = Field(None, description="Reranker model ID")
    is_dev: bool = Field(
        ..., description="Whether running in development mode"
    )
