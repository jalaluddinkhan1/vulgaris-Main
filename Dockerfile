FROM python:3.11-slim

# Sane Python defaults for containers
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# No build-essential needed — VULGARIS is pure Python / NumPy
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (layer-cached when source changes but deps don't)
COPY pyproject.toml README.md LICENSE ./
RUN pip install ".[serve]"

# Copy source (invalidates cache only when code changes)
COPY . .
RUN pip install --no-deps -e .

# Non-root user for security
RUN useradd --no-log-init --system --uid 1001 vulgaris \
    && chown -R vulgaris:vulgaris /app
USER vulgaris

# Model configuration (override at runtime via env or docker-compose)
ENV VULGARIS_INPUT_DIM=9 \
    VULGARIS_OUTPUT_DIM=1 \
    VULGARIS_N_CLASSES=5 \
    VULGARIS_D_MODEL=256 \
    VULGARIS_N_DOMAINS=32 \
    VULGARIS_RMC_EXPERTS=4 \
    VULGARIS_API_KEYS="" \
    VULGARIS_LOG_LEVEL=INFO \
    VULGARIS_CHECKPOINT="" \
    VULGARIS_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["python", "-m", "inference.server_entrypoint"]
