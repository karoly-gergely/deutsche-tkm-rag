"""
FastAPI application initialization and configuration.
Handles app creation, middleware setup, and lifespan management.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.utils.dependencies import (
    get_model,
    get_prompt_manager,
    get_retriever,
    get_tokenizer,
    get_vectordb,
)
from config import settings
from monitoring.tracing import get_tracer, setup_tracing

logger = logging.getLogger(__name__)


# Initialize tracing
setup_tracing()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle warm-up on startup."""
    logger.info("Warming up components...")
    get_tokenizer()
    get_model()
    get_vectordb()
    get_retriever()
    get_prompt_manager()
    logger.info("✅ Warm-up complete")

    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app instance.
    """
    # Create FastAPI app
    app = FastAPI(
        title="RAG Assistant API", version="1.0.0", lifespan=lifespan
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # OpenTelemetry middleware if enabled
    if settings.ENABLE_TRACING:
        tracer = get_tracer()
        if tracer is not None:
            try:
                from opentelemetry.instrumentation.fastapi import (
                    FastAPIInstrumentor,
                )

                FastAPIInstrumentor.instrument_app(app)
            except ImportError:
                # OpenTelemetry FastAPI instrumentation not available
                pass

    return app


# Create the app instance
app = create_app()
