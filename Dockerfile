FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Environment
ENV VULGARIS_INPUT_DIM=9
ENV VULGARIS_N_CLASSES=5
ENV VULGARIS_D_MODEL=64
ENV VULGARIS_API_KEYS=""
ENV VULGARIS_LOG_LEVEL=INFO
ENV VULGARIS_CHECKPOINT=""
ENV VULGARIS_PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${VULGARIS_PORT}/health')"

EXPOSE 8000

CMD ["python", "-m", "inference.server_entrypoint"]
