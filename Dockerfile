# Optimized Dockerfile for Clinical Summary RAG Application
# Multi-stage build to reduce image size
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy production-optimized requirements (smaller image)
COPY ingestion-phase/requirements-prod.txt /tmp/requirements.txt

# Install Python dependencies
# First install regular packages from PyPI (system-wide for reliability)
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Then install CPU-only PyTorch (much smaller than GPU version)
# This reduces torch from ~2GB to ~200MB
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch>=2.0.0

# Install transformers and huggingface (after torch)
RUN pip install --no-cache-dir transformers>=4.30.0 huggingface-hub>=0.16.0

# Verify transformers installation
RUN python -c "import transformers; print(f'Transformers version: {transformers.__version__}')" || echo "ERROR: Transformers not installed!"

# Clean up pip cache
RUN pip cache purge

# Final stage - minimal runtime image
FROM python:3.10-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy Python packages from builder (system-wide installation)
# Copy site-packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only necessary application files (excludes large files via .dockerignore)
COPY ingestion-phase/api.py ingestion-phase/app.py ingestion-phase/config.py ./
COPY ingestion-phase/run_app.py ingestion-phase/setup.py ./
# Copy start_services.sh explicitly (must not be in .dockerignore)
COPY ingestion-phase/start_services.sh ./start_services.sh

# Copy scripts directory (excluding notebooks via .dockerignore)
COPY ingestion-phase/scripts/ ./scripts/

# Create necessary directories (will be populated at runtime)
# Vector DB and embeddings are generated at runtime, NOT included in image
RUN mkdir -p vector_db/chroma embeddings processed data logs temp .cache/transformers .cache/huggingface && \
    # Clean up any accidentally copied large files
    rm -rf vector_db/* embeddings/* processed/* data/* 2>/dev/null || true

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8000 8501

# Health check - check Streamlit (main service)
# Railway will check the PORT environment variable
# Use a longer start period for model loading (3 minutes)
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -f http://localhost:8501/_stcore/health || curl -f http://localhost:8501/ || exit 1

# Make startup script executable
RUN chmod +x /app/start_services.sh

# Start both services
CMD ["/app/start_services.sh"]
