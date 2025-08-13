#!/bin/bash
set -e

echo "🔄 Starting RAG Chatbot..."

export PYTHONPATH=/app

PORT=$(python -c "from app.config import load_config; print(load_config().get('PORT', 8080))")
echo "✅ Using port: $PORT"

exec python -m app.main
