# RAG Assistant

A production-ready Retrieval-Augmented Generation (RAG) system for building intelligent document-based question-answering applications.

## Features

- **Document Processing**: Support for PDF, TXT, and JSON formats
- **Vector Store**: ChromaDB integration for efficient similarity search
- **Reranking**: Cross-encoder models for improved retrieval accuracy
- **Streaming**: Real-time response generation with token streaming
- **API**: FastAPI-based REST API for integration
- **UI**: Streamlit-based web interface
- **Monitoring**: Logging, metrics, and OpenTelemetry tracing support

## Architecture

```
┌─────────────┐
│   User      │
└─────┬───────┘
      │
      ├─────────────────┬─────────────────┐
      │                 │                 │
      ▼                 ▼                 ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Streamlit│    │ FastAPI  │    │ Scripts  │
│   UI     │    │   API    │    │ (CLI)    │
└────┬─────┘    └────┬─────┘    └────┬─────┘
     │              │                │
     └──────────────┼────────────────┘
                    │
                    ▼
          ┌──────────────────┐
          │  LLM Generator   │
          │  (Qwen2.5-3B)    │
          └─────────┬────────┘
                    │
                    ▼
          ┌──────────────────┐
          │ Retrieval Engine │
          │  (ChromaDB)      │
          └─────────┬────────┘
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
    ┌──────────┐      ┌──────────────┐
    │Embeddings│      │  Chunking    │
    │  Model   │      │  Strategy    │
    └──────────┘      └──────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip or conda

### Installation

1. **Clone the repository** (or navigate to project directory):
   ```bash
   cd /path/to/project
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Ingest Documents

Place your documents (PDF, TXT, JSON) in the `data/` folder, then run:

```bash
python scripts/ingest.py
```

Or specify a custom data folder:

```bash
python scripts/ingest.py --data-folder /path/to/documents
```

### Run the Streamlit UI

```bash
streamlit run ui/streamlit_app.py
```

The application will be available at `http://localhost:8501`

### Run the API Server

```bash
uvicorn api.routes:app --reload --host 0.0.0.0 --port 8000
```

API documentation will be available at `http://localhost:8000/docs`

## Configuration

All configuration is managed through environment variables. See `.env.example` for available options:

- `MODEL_ID`: LLM model identifier (default: Qwen/Qwen2.5-3B-Instruct)
- `EMBEDDING_MODEL`: Embedding model for vectorization
- `DATA_FOLDER`: Path to documents folder
- `CHROMA_DIR`: ChromaDB persistence directory
- `CHUNK_SIZE`: Text chunk size in words
- `TOP_K`: Number of retrieved documents
- `MAX_NEW_TOKENS`: Maximum tokens to generate
- `TEMPERATURE`: Sampling temperature
- And more...

## Project Structure

```
project/
├── config/          # Configuration management
├── core/            # Core RAG functionality (embeddings, chunking, retrieval)
├── data/            # Data loading and processing
├── llm/             # LLM management and generation
├── monitoring/      # Logging, metrics, tracing
├── api/             # FastAPI routes
├── ui/              # Streamlit interface
├── tests/           # Unit tests
└── scripts/         # Utility scripts
```

## Development

### Linting and Formatting

This project uses `black`, `isort`, and `ruff` for code quality:

```bash
black .
isort .
ruff check .
```

Configuration is in `pyproject.toml`.

### Running Tests

```bash
pytest
```

### Rebuilding the Index

To rebuild the vector store from scratch:

```bash
python scripts/rebuild_index.py --force
```

## Docker

### Build and Run with Docker Compose

```bash
docker-compose up --build
```

This will:
- Build the application image
- Start the API server on port 8000
- Start the Streamlit UI on port 8501

### Dockerfile Only

```bash
docker build -t rag-assistant .
docker run -p 8000:8000 rag-assistant
```

## Production Checklist

Before deploying to production, ensure the following security and hardening measures are implemented:

### 🔐 Secrets & Configuration

- [ ] **Move secrets to environment/secret store**: Never commit API keys, tokens, or credentials. Use a secret management service (AWS Secrets Manager, HashiCorp Vault, Azure Key Vault, etc.)
- [ ] **Rotate secrets regularly**: Implement a rotation policy for API keys, database passwords, and authentication tokens
- [ ] **Review `.env` files**: Ensure `.env` is in `.gitignore` and never committed to version control
- [ ] **Use separate configs**: Maintain separate configuration for development, staging, and production environments
- [ ] **Poetry lockfile integrity**: Ensure `poetry.lock` is committed and consistent with `pyproject.toml` to guarantee reproducible builds
- [ ] **Environment schema validation**: Add a startup check using Pydantic BaseSettings to validate required environment variables are present (e.g. `MODEL_ID`, `DATA_FOLDER`, `CHROMA_DIR`)

### 📊 Monitoring & Observability

- [ ] **Enable tracing**: Set `ENABLE_TRACING=true` and configure OpenTelemetry exporters to send traces to an OTEL collector
- [ ] **Metrics aggregation**: Instrument metrics (latency, error rates, token usage) and export to Prometheus/Grafana or similar
- [ ] **Centralized logging**: Configure log aggregation to ELK stack, OpenSearch, or cloud logging service (CloudWatch, Stackdriver)
- [ ] **Alerting**: Set up alerts for error rates, latency spikes, cost overruns, and system health issues
- [ ] **Dashboard**: Create monitoring dashboards for real-time visibility into system performance
- [ ] **Startup health verification**: Automatically run `curl -f http://localhost:8080/healthz` and `curl -f http://localhost:8501/?health=true` as part of deployment health checks

### 🛡️ API Security & Rate Limiting

- [ ] **Request/response size caps**: Configure maximum request and response sizes in FastAPI/uvicorn:
  ```python
  # In uvicorn config
  --limit-concurrency 100
  --timeout-keep-alive 5
  # In FastAPI
  app = FastAPI(max_request_size=1024*1024)  # 1MB limit
  ```
- [ ] **Timeouts**: Set appropriate timeouts for:
  - HTTP requests (client timeout)
  - LLM generation (max generation time)
  - Database queries
  - External API calls
- [ ] **Rate limiting**: Implement rate limiting per user/IP to prevent abuse:
  ```python
  from slowapi import Limiter
  limiter = Limiter(key_func=get_remote_address)
  ```
- [ ] **Input validation**: Add Pydantic schemas for all API endpoints, especially `/query`:
  ```python
  class QueryRequest(BaseModel):
      query: str = Field(..., min_length=1, max_length=1000)
      top_k: Optional[int] = Field(None, ge=1, le=50)
  ```

### 🔍 Retrieval Enhancements

- [ ] **MMR Retriever**: Implement Max Marginal Relevance to reduce redundant results:
  ```python
  from langchain.retrievers import MMRRetriever
  ```
- [ ] **Duplicate suppression**: Add deduplication logic to filter out near-duplicate chunks before returning results
- [ ] **Answer citations**: Enhance citations with line/character span references where possible for better traceability
- [ ] **A/B testing**: Set up A/B testing framework for reranker models; log per-query feature flags to analyze performance

### 🧪 Testing & Evaluation

- [ ] **Offline evaluation**: Implement RAGAS (Retrieval-Augmented Generation Assessment) evaluation harness:
  - Context Precision
  - Context Recall
  - Faithfulness
  - Answer Relevancy
- [ ] **Golden set regression**: Maintain a golden set of queries and expected outputs; run regression tests on model changes
- [ ] **Automated testing**: Ensure CI/CD pipeline runs full test suite including integration tests
- [ ] **Performance benchmarks**: Establish baseline metrics for latency, throughput, and accuracy

### 🐳 Container Security

- [ ] **Non-root user**: Update Dockerfile to run as non-root user:
  ```dockerfile
  RUN useradd -m -u 1001 appuser
  USER 1001
  ```
- [ ] **Minimal base image**: Use minimal base images (python:3.11-slim) and remove unnecessary packages
- [ ] **Multi-stage build optimization**: Use a two-stage Docker build (builder → runtime) to reduce final image size and attack surface
- [ ] **Read-only filesystem**: Add `read_only: true` in production Compose deployment for immutable container filesystems
- [ ] **Security scanning**: Run vulnerability scans on container images:
  ```bash
  # Using trivy
  trivy image rag-assistant
  # Using pip-audit for Python dependencies
  pip-audit
  ```
- [ ] **CI vulnerability checks**: Integrate `pip-audit` and `trivy` into CI pipeline to block vulnerable images

### ⚡ Performance & Caching

- [ ] **Caching layer**: Implement caching for frequent queries using SQLite, DuckDB, or Redis:
  ```python
  # Cache query results with TTL
  from functools import lru_cache
  from cachetools import TTLCache
  ```
- [ ] **Embedding cache**: Cache document embeddings to avoid recomputation
- [ ] **Response cache**: Cache responses for identical queries (with appropriate invalidation strategy)
- [ ] **Database optimization**: Optimize ChromaDB queries and index maintenance
- [ ] **Persistent volume checks**: Verify `data/` and `.chroma/` volumes exist and are writable before application startup

### 🔒 Additional Security Measures

- [ ] **HTTPS/TLS**: Ensure all API endpoints use HTTPS in production
- [ ] **Authentication/Authorization**: Implement proper auth (OAuth2, JWT, API keys) based on your requirements
- [ ] **CORS configuration**: Restrict CORS origins to known domains
- [ ] **Content filtering**: Add input/output content filtering to prevent prompt injection and sensitive data leakage
- [ ] **Audit logging**: Log all queries, responses, and system actions for compliance and debugging

### 📋 CI/CD Pipeline

- [ ] **Automated builds**: Set up CI to build and test on every commit
- [ ] **Poetry install verification**: Include a CI step to run `poetry check && poetry install --no-root --no-interaction` to validate dependency integrity before build
- [ ] **Dependency caching**: Cache Poetry dependencies in CI to speed up builds
- [ ] **Vulnerability scanning**: Run `pip-audit` and `trivy` scans as part of CI pipeline
- [ ] **Automated deployment**: Configure automated deployment to staging/production with manual approval gates
- [ ] **Rollback strategy**: Establish rollback procedures for failed deployments

### 📝 Documentation

- [ ] **Runbooks**: Create operational runbooks for common issues and recovery procedures
- [ ] **API documentation**: Ensure OpenAPI/Swagger documentation is up-to-date
- [ ] **Architecture diagrams**: Document production architecture, data flows, and dependencies
- [ ] **Incident response**: Document incident response procedures and escalation paths

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Notes

### Telemetry

OpenTelemetry tracing is disabled by default. To enable:

1. Set `ENABLE_TRACING=true` in `.env`
2. Configure exporters in `monitoring/tracing.py`

### Model Licensing

- **Qwen2.5-3B-Instruct**: Apache 2.0 License (commercial use allowed)
- **sentence-transformers/all-MiniLM-L6-v2**: Apache 2.0 License
- **cross-encoder/ms-marco-MiniLM-L-6-v2**: Apache 2.0 License

Please review individual model licenses before commercial deployment.

## Troubleshooting

### Import Errors

If you encounter import errors, ensure you're running from the project root and the virtual environment is activated.

### ChromaDB Issues

If ChromaDB fails to initialize:
1. Check write permissions for `CHROMA_DIR`
2. Delete `.chroma/` and rebuild the index

### Model Loading Issues

For large models:
- Ensure sufficient RAM/VRAM
- Use quantization (configured for CPU/4-bit)
- Consider using smaller models for development

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure linting passes
5. Submit a pull request

