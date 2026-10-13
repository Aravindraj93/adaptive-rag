# syntax=docker/dockerfile:1
FROM python:3.11-slim AS runtime

# Set security and Python runtime environments
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r raguser && useradd -r -g raguser -d /app -s /sbin/nologin raguser

# Install adaptive-rag-py
COPY . /app
RUN pip install --no-cache-dir .

# Create /data directory for mounting document corpus
RUN mkdir -p /data && chown -R raguser:raguser /app /data

USER raguser

EXPOSE 8000

# Built-in health check using /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["adaptive-rag"]
CMD ["serve", "--corpus", "/data", "--host", "0.0.0.0", "--port", "8000"]
