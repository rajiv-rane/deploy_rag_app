# Optimized Dockerfile for Clinical Summary RAG Application
# Multi-stage build to reduce image size
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY ingestion-phase/requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt

# Final stage - minimal runtime image
FROM python:3.10-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy only necessary application files (excludes large files via .dockerignore)
COPY ingestion-phase/api.py ingestion-phase/app.py ingestion-phase/config.py ./
COPY ingestion-phase/start_services.sh ./
COPY ingestion-phase/run_app.py ingestion-phase/setup.py ./

# Copy scripts directory (excluding notebooks via .dockerignore)
COPY ingestion-phase/scripts/ ./scripts/

# Create necessary directories (will be populated at runtime)
RUN mkdir -p vector_db/chroma embeddings processed data logs temp .cache/transformers .cache/huggingface

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Make startup script executable
RUN chmod +x /app/start_services.sh

# Start both services
CMD ["/app/start_services.sh"]
