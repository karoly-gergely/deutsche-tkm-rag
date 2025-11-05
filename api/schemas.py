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
