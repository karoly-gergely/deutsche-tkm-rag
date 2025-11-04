FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies (including curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    python3.11 python3.11-venv python3-pip \
    && ln -s /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.8.3
RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

# Configure Poetry: disable virtualenvs, non-interactive mode
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    PYTHONPATH=/app

# Build argument to control dev dependency installation
ARG INSTALL_DEV_DEPS=false

# Copy Poetry configuration files first for better layer caching
COPY pyproject.toml ./
# Copy poetry.lock if it exists (will be included in COPY . . below if present)
# If missing, Poetry will generate it during install

# Install dependencies conditionally based on build arg
# Production: --no-dev excludes dev dependencies
# Development: includes dev dependencies (accelerate, etc.)
RUN if [ "$INSTALL_DEV_DEPS" = "true" ]; then \
        poetry install --no-ansi; \
    else \
        poetry install --no-dev --no-ansi; \
    fi && \
    rm -rf $POETRY_CACHE_DIR

# Copy application code
COPY . .

# Copy entrypoint script
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# GPU env
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="8.0"

# Expose ports
EXPOSE 8501 8080

# Default command: run Streamlit UI via Poetry and FastAPI backend
CMD ["./entrypoint.sh"]
