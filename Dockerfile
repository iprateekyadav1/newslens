# ── Stage 1: dependency installation ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install
COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/deps -r requirements.txt


# ── Stage 2: final runtime image ──────────────────────────────────────────────
FROM python:3.11-slim

LABEL org.opencontainers.image.title="NewsLens Bias Detector"
LABEL org.opencontainers.image.description="Media bias detection API + dashboard"

# Non-root user for security
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /deps /usr/local

# Copy application code
COPY --chown=appuser:appuser . .

# Ensure model / log directories exist
RUN mkdir -p models logs data/raw data/processed

USER appuser

EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/api/health')"

# Production server: 2 workers, 2 threads each
CMD ["gunicorn", "app.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:5001", \
     "--timeout", "120", \
     "--access-logfile", "-"]
