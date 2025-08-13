# ==========================
# Dockerfile (CPU-friendly)
# ==========================
FROM python:3.11-slim

WORKDIR /app

# ----------------------------------------------------
# Install system dependencies
# ----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------
# Install Python dependencies
# ----------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------
# Copy application code
# ----------------------------------------------------
COPY app /app/app

# Copy the full Gemma model into the container
COPY app/models/gemma /app/app/models/gemma

# ----------------------------------------------------
# Prebuild embedder pickle inside image
RUN python - <<'PY'
from app.config import load_config
from app.retriever import load_embedder
cfg = load_config()
load_embedder(cfg)
PY

# ----------------------------------------------------
# Copy entrypoint script
# ----------------------------------------------------
COPY docker-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# ----------------------------------------------------
# Expose API port
# ----------------------------------------------------
EXPOSE 8080

ENTRYPOINT ["entrypoint.sh"]
