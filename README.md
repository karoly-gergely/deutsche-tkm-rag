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
                     │    User     │
                     └──────┬──────┘
                            │
      ┌──────────┬──────────┼──────────┬──────────┐
      │          │          │          │          │
      │          │          │          │          │
      ▼          ▼          ▼          ▼          ▼
    ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    │Streamlit│ │React App │ │ FastAPI  │ │ Scripts  │
    │   UI    │ │          │ │   API    │ │  (CLI)   │
    └────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
         │           │            │            │
         │           └────────────┘            │
         │           (uses FastAPI)            │
         │                 │                   │
         │                 │                   │
         └─────────────────┼───────────────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  LLM Generator   │
                  │  (Gemma-2-9B)    │
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

- Python 3.11 or higher
- [Poetry](https://python-poetry.org/docs/#installation) for Python dependency management
- Node.js 18+ and pnpm (for React frontend)

### Installation

1. **Clone the repository** (or navigate to project directory):
   ```bash
   cd /path/to/project
   ```

2. **Install Python dependencies with Poetry**:
   ```bash
   poetry install
   ```

3. **Set up environment variables**:
   ```bash
   make setup
   # Or manually: cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Install the project** (required for CLI commands):
   
   After installing dependencies, the project is automatically installed as a package. This enables Poetry CLI commands to work without `PYTHONPATH` configuration.
   
   ```bash
   poetry install
   ```
   
   This installs the project packages (`config`, `core`, `api`, etc.) and registers CLI entry points (`ingest`, `rebuild-index`, `dev-check`, etc.).

5. **Set up React frontend** (optional):
   ```bash
   make react-setup    # Creates .env file in react/ directory
   make react-install  # Installs pnpm and React dependencies
   ```
   
   Or run directly (from 'react' folder):
   ```bash
   cp .env.example .env                # Creates .env file in react/ directory
   npm install -g pnpm                 # Install pnpm globally
   pnpm install                        # Install React dependencies
   ```

### Ingest Documents

Place your documents (TXT files) in the `data/` folder, then run:

```bash
make ingest
```

Or using Poetry CLI directly:

```bash
poetry run ingest
```

To specify a custom data folder:

```bash
poetry run ingest --data-folder /path/to/documents
```

### Run the Streamlit UI

```bash
make ui
```

Or using Poetry directly:

```bash
poetry run streamlit run ui/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

The application will be available at `http://localhost:8501`

### Run the API Server

```bash
make api
```

Or using Poetry directly:

```bash
poetry run uvicorn api:app --reload --host 0.0.0.0 --port 8080
```

API documentation will be available at `http://localhost:8080/docs`

### Run the React Frontend

```bash
make react-dev
```

Or run directly (from 'react' folder):
```bash
pnpm dev
```

The React application will be available at `http://localhost:5173`

Other React commands:
- `make react-build` - Build for production
  
  Or run directly (from 'react' folder):
  ```bash
  pnpm build
  ```
- `make react-lint` - Run linting
  
  Or run directly (from 'react' folder):
  ```bash
  pnpm lint
  ```
- `make react-preview` - Preview production build
  
  Or run directly (from 'react' folder):
  ```bash
  pnpm preview
  ```

## Configuration

### Backend Configuration

All backend configuration is managed through environment variables. See `.env.example` for available options:

- `MODEL_ID`: LLM model identifier (default: `google/gemma-2-9b-it`)
- `DEV_MODEL_ID`: LLM model identifier for running in DEV mode (default: `google/gemma-2-2b-it`)
- `EMBEDDING_MODEL`: Embedding model for vectorization (default: `intfloat/multilingual-e5-large`)
- `DEVICE`: Device for model execution, `cuda` or `cpu` (default: `cuda`)
- `DATA_FOLDER`: Path to documents folder (default: `data`)
- `CHROMA_DIR`: ChromaDB persistence directory (default: `.chroma`)
- `CHUNK_SIZE`: Text chunk size in characters (default: `800`)
- `CHUNK_OVERLAP`: Overlap between chunks in characters (default: `150`)
- `TOP_K`: Number of retrieved documents (default: `5`)
- `RERANK_TOP_K`: Number of candidates for reranking (default: `10`)
- `RERANKER_MODEL`: Reranker model identifier (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `MAX_CONTEXT_TOKENS`: Maximum context tokens (default: `6000`)
- `MAX_NEW_TOKENS`: Maximum tokens to generate (default: `768`)
- `TEMPERATURE`: Sampling temperature (default: `0.6`)
- `TOP_P`: Nucleus sampling parameter (default: `0.9`)
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `ENABLE_TRACING`: Enable OpenTelemetry tracing (default: `false`)
- `LOG_DIR`: Log directory (default: `logs`)

### React Frontend Configuration

React environment variables are configured in `react/.env` (created via `make react-setup`). Available variables:

- `VITE_BASENAME`: Base path for the React app when deployed to a subdirectory (default: `/deutsche-tkm-rag`)
  - Used in production builds to set the router basename and Vite base path
  - In development, always uses `/`
  
- `VITE_API_URL`: API base URL for the backend (default: `http://localhost:8080`)
  - Should point to your FastAPI backend
  - For production, use the full URL (e.g., `https://api.example.com`)

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

Using Makefile:
```bash
make test
```

Or using Poetry directly:
```bash
poetry run pytest
```

### Rebuilding the Index

To rebuild the vector store from scratch:

Using Makefile:
```bash
make rebuild-index-force
```

Or using Poetry CLI directly:
```bash
poetry run rebuild-index --force
```

## Docker

### Build and Run with Docker Compose (everything besides the React Application)

```bash
docker-compose -f docker/docker-compose.yml up --build
```

Or use the Makefile:
```bash
make up
```

This will:
- Build the application image
- Start the API server on port 8080
- Start the Streamlit UI on port 8501

### Development Mode (CPU-only, hot-reload) (everything besides the React Application)

```bash
make dev
# or
docker-compose -f docker/docker-compose.dev.yml up --build
```

### Dockerfile Only (everything besides the React Application)

```bash
docker build -f docker/Dockerfile -t rag-assistant .
docker run -p 8080:8080 -p 8501:8501 rag-assistant
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

- **Gemma-2-9B-IT**: Apache 2.0 License (commercial use allowed)
- **Gemma-2-2B-IT**: Apache 2.0 License (commercial use allowed)
- **intfloat/multilingual-e5-large**: Apache 2.0 License (commercial use allowed)
- **cross-encoder/ms-marco-MiniLM-L-6-v2**: Apache 2.0 License (commercial use allowed)

Please review individual model licenses before commercial deployment.

## Troubleshooting

### Import Errors

If you encounter import errors:
1. Ensure the project is installed: `poetry install` (installs project as package)
2. Use Poetry CLI commands (`poetry run ingest`, `poetry run rebuild-index`, etc.)
3. Ensure you're running from the project root
4. Verify dependencies are installed: `poetry install`

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

