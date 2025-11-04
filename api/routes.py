"""FastAPI routes for RAG application."""
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.dependencies import get_tokenizer, get_model, get_vectordb, get_retriever, get_prompt_manager, get_logger
from config import settings, IS_DEV
from llm.generation import generate_response
from monitoring.metrics import get_metrics_registry
from monitoring.tracing import get_tracer, setup_tracing


# Initialize tracing
setup_tracing()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle warm-up on startup."""
    logger = get_logger()
    logger.info("Warming up components...")
    get_tokenizer()
    get_model()
    get_vectordb()
    get_retriever()
    get_prompt_manager()
    logger.info("✅ Warm-up complete")
    
    yield

# Create FastAPI app
app = FastAPI(title="RAG Assistant API", version="1.0.0", lifespan=lifespan)

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
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
        except ImportError:
            # OpenTelemetry FastAPI instrumentation not available
            pass


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    top_k: Optional[int] = Field(
        None, ge=1, le=20, description="Number of documents to retrieve (clamped to 1-20)"
    )


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    answer: str
    sources: List[str]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    sizes: Dict[str, Optional[int]]


@app.get("/health")
async def health_simple():
    """Simple health check endpoint."""
    from fastapi.responses import JSONResponse
    return JSONResponse(content={"status": "ok"}, status_code=200)


@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system sizes."""
    sizes = {}
    num_vectors = None

    try:
        vectordb = get_vectordb()
        # Try to get collection info
        if hasattr(vectordb, "_collection"):
            collection = vectordb._collection
            if collection is not None:
                count_result = collection.count()
                if count_result is not None:
                    num_vectors = int(count_result)
    except Exception:
        # If vector store is not available or count fails, leave as None
        pass

    sizes["num_vectors"] = num_vectors

    return HealthResponse(status="ok", sizes=sizes)


@app.get("/metrics")
async def metrics():
    """Temporary metrics endpoint returning JSON counters.
    
    Returns:
        JSON dictionary with counters and timers from metrics registry.
    """
    registry = get_metrics_registry()
    # Format for JSON response
    result = {
        "counters": registry.get("counters", {}),
        "timers": {
            name: {
                "count": len(timings),
                "avg_ms": sum(timings) / len(timings) if timings else 0,
                "min_ms": min(timings) if timings else 0,
                "max_ms": max(timings) if timings else 0,
            }
            for name, timings in registry.get("timers", {}).items()
        },
    }
    return result


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system."""
    logger = get_logger()
    start_time = time.time()

    try:
        # Get components
        retriever = get_retriever()
        prompt_manager = get_prompt_manager()
        tokenizer = get_tokenizer()
        model = get_model()

        # Determine top_k (clamped to [1, 20])
        top_k = request.top_k or settings.TOP_K
        top_k = max(1, min(20, top_k))  # Ensure it's in valid range

        # Retrieve documents
        retrieved_docs = retriever.retrieve(query=request.query, top_k=top_k)

        if not retrieved_docs:
            raise HTTPException(
                status_code=404, detail="No relevant documents found"
            )

        # Build prompt
        prompt = prompt_manager.build_rag_prompt(
            query=request.query, context_docs=retrieved_docs, chat_history=None
        )

        # Adaptive generation parameters based on dev mode
        if IS_DEV:
            max_tokens = 128
            temperature = 0.8
            top_p = 0.9
        else:
            max_tokens = settings.MAX_NEW_TOKENS
            temperature = settings.TEMPERATURE
            top_p = settings.TOP_P

        # Generate response (use threadpool for non-blocking execution)
        from starlette.concurrency import run_in_threadpool

        def _generate():
            return generate_response(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                device=settings.DEVICE,
            )

        answer = await run_in_threadpool(_generate)

        # Extract publication IDs from sources
        publication_ids = []
        for doc in retrieved_docs:
            pub_id = doc.metadata.get(
                "publication_id", doc.metadata.get("doc_id", "unknown")
            )
            if pub_id not in publication_ids:
                publication_ids.append(pub_id)

        # Calculate response time
        response_time = time.time() - start_time

        # Log query
        logger.log_query(
            query=request.query,
            retrieved_docs=retrieved_docs,
            response_time=response_time,
        )

        return QueryResponse(answer=answer, sources=publication_ids)

    except HTTPException:
        raise
    except Exception as e:
        logger.log_error(error=e, context={"query": request.query})
        raise HTTPException(status_code=500, detail=str(e))

