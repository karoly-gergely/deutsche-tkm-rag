FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.8.3
RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

# Configure Poetry: disable virtualenvs, non-interactive mode
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Copy Poetry configuration files first for better layer caching
COPY pyproject.toml ./
# Copy poetry.lock if it exists (will be included in COPY . . below if present)
# If missing, Poetry will generate it during install

# Install dependencies (will generate poetry.lock if missing)
# Install package in editable mode so imports work correctly
RUN poetry install --no-ansi && rm -rf $POETRY_CACHE_DIR

# Copy application code
COPY . .

# Expose ports
EXPOSE 8501 8080

# Default command: run Streamlit UI via Poetry
CMD ["poetry", "run", "streamlit", "run", "ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
