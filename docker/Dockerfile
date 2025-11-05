FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies (including curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    python3.10 python3.10-venv python3-pip python3.10-dev \
    gcc g++ \
    && python3.10 -m pip install --upgrade pip setuptools wheel \
    && ln -sf $(which python3.10) /usr/local/bin/python \
    && ln -sf $(which pip3) /usr/local/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Fix bitsandbytes/triton linking error: "cannot find -lcuda"
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so

# Install Poetry
ENV POETRY_VERSION=1.8.3
RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

# Configure Poetry: disable virtualenvs, non-interactive mode
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Build argument to control dev dependency installation
ARG INSTALL_DEV_DEPS=false

# Copy Poetry configuration files first for better layer caching
COPY pyproject.toml ./

# Install dependencies conditionally based on build arg
# Production: --no-dev excludes dev dependencies
# Development: includes dev dependencies (accelerate, etc.)
# Use --no-root to skip installing the project itself (code is copied later)
RUN if [ "$INSTALL_DEV_DEPS" = "true" ]; then \
        poetry install --no-root --no-ansi; \
    else \
        poetry install --no-root --no-dev --no-ansi; \
    fi && \
    rm -rf $POETRY_CACHE_DIR

# Copy application code
COPY . .

RUN rm -f poetry.lock

# Install the project package (enables imports without PYTHONPATH)
# --only-root because dependencies were already installed above
RUN poetry install --only-root --no-ansi

## SSL
RUN mkdir -p /app/certs && \
    openssl req -x509 -newkey rsa:4096 \
      -keyout /app/certs/key.pem \
      -out /app/certs/cert.pem \
      -days 365 -nodes \
      -subj "/CN=localhost"

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
