# ============================================================================
# ShivX AI Trading System - Production Dockerfile
# ============================================================================
# Multi-stage build for optimized production image
# ============================================================================

# Stage 1: Builder
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 2: Runtime
FROM python:3.11-slim

# Metadata
LABEL maintainer="dev@shivx.ai"
LABEL version="2.0.0"
LABEL description="ShivX AI Trading System"

# Create non-root user
RUN groupadd -r shivx && useradd -r -g shivx shivx

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ ./app/
COPY core/ ./core/
COPY utils/ ./utils/
COPY security/ ./security/
COPY config/ ./config/
COPY observability/ ./observability/
COPY main.py .
COPY pyproject.toml .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models var/secrets && \
    chown -R shivx:shivx /app

# Switch to non-root user
USER shivx

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health/live || exit 1

# Environment variables (defaults, override with env file)
ENV SHIVX_ENV=production
ENV SHIVX_HOST=0.0.0.0
ENV SHIVX_PORT=8000
ENV PYTHONUNBUFFERED=1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
