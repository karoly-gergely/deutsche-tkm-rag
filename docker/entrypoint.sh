#!/bin/bash
set -e

# Run ingestion if ChromaDB needs it
if poetry run check-needs-ingestion; then
    echo "✓ Vector store already exists with data, skipping ingestion"
else
    echo "📚 Running ingestion..."
    poetry run ingest
fi

# Start FastAPI backend
echo "🚀 Starting FastAPI backend..."
poetry run uvicorn api.routes:app --host 0.0.0.0 --port 8080 --ssl-keyfile /app/certs/key.pem --ssl-certfile /app/certs/cert.pem &

# Start Streamlit UI
echo "🎨 Starting Streamlit frontend..."
poetry run streamlit run ui/streamlit_app.py --server.port=8501 --server.address=0.0.0.0 &

# Wait for both processes
wait -n
