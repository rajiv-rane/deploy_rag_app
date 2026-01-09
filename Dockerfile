# Dockerfile for Clinical Summary RAG Application
# This Dockerfile is in the root and builds from ingestion-phase directory
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY ingestion-phase/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code from ingestion-phase
COPY ingestion-phase/ .

# Create necessary directories
RUN mkdir -p vector_db/chroma embeddings processed data logs temp

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV HF_HOME=/app/.cache/huggingface

# Expose ports
# Port 8000 for FastAPI, Port 8501 for Streamlit
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use the startup script
RUN chmod +x /app/start_services.sh

# Start both services
CMD ["/app/start_services.sh"]
