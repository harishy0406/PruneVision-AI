# ─────────────────────────────────────────────────────────────────────────────
# PruneVision AI - Multi-Stage Production Dockerfile
# Optimized for model serving and CPU inference
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime
FROM python:3.11-slim

# Security: Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local

# Set environment variables
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/app/.cache/torch \
    TORCHVISION_HOME=/app/.cache/torchvision

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/outputs/{checkpoints,exports,logs} \
    && mkdir -p /app/.cache/torch \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import streamlit; print('OK')" || exit 1

# Expose port for Streamlit
EXPOSE 8501

# Default command: Run Streamlit app
CMD ["streamlit", "run", "app_advanced.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--logger.level=info"]

# ─────────────────────────────────────────────────────────────────────────────
# Build metadata
LABEL maintainer="PruneVision AI Team" \
      description="PruneVision AI - Neural Network Pruning Dashboard" \
      version="1.0.0"
