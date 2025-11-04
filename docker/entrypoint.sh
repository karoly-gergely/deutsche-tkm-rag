#!/bin/bash
set -e

# Start FastAPI backend
echo "🚀 Starting FastAPI backend..."
poetry run uvicorn api.routes:app --host 0.0.0.0 --port 8080 &

# Start Streamlit UI
echo "🎨 Starting Streamlit frontend..."
poetry run streamlit run ui/streamlit_app.py --server.port=8501 --server.address=0.0.0.0 &

# Wait for both processes
wait -n
