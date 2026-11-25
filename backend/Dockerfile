# ── Stage 1: build / dependency install ──────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-test.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create data directory
RUN mkdir -p /data

# Copy application code
COPY app/ ./app/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app /data
USER appuser

EXPOSE 8000

ENV DATABASE_URL=sqlite:////data/app.db \
    CACHE_DIR=/data/litellm_cache \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
