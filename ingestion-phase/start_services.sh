#!/bin/bash
# Startup script to run both FastAPI and Streamlit services
# More robust version that handles errors gracefully

echo "============================================================"
echo "🚀 Starting Clinical Summary RAG Application"
echo "============================================================"

# Check if GROQ_API_KEY is set
if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  WARNING: GROQ_API_KEY environment variable is not set!"
    echo "   The application may not work correctly."
else
    echo "✅ GROQ_API_KEY is set"
fi

# Verify critical Python packages are installed
echo "🔍 Verifying Python packages..."
python -c "import transformers; print(f'✅ Transformers {transformers.__version__} installed')" 2>/dev/null || echo "❌ ERROR: Transformers not found!"
python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed')" 2>/dev/null || echo "❌ ERROR: PyTorch not found!"
python -c "import streamlit; print(f'✅ Streamlit {streamlit.__version__} installed')" 2>/dev/null || echo "❌ ERROR: Streamlit not found!"
python -c "import fastapi; print(f'✅ FastAPI {fastapi.__version__} installed')" 2>/dev/null || echo "❌ ERROR: FastAPI not found!"
echo ""

# Get port from environment variable (Railway/Render sets this)
# Use PORT for Streamlit (main service), FastAPI on internal port
EXTERNAL_PORT=${PORT:-8501}
FASTAPI_PORT=${FASTAPI_PORT:-8000}

# For Railway/Render: PORT is the external port, use internal port for FastAPI
# FastAPI will run on 8000 internally, Streamlit on the assigned PORT
echo "📍 FastAPI will run on port: $FASTAPI_PORT (internal)"
echo "📍 Streamlit will run on port: $EXTERNAL_PORT (external)"
echo "============================================================"

# Set FastAPI URL for Streamlit to connect
export FASTAPI_URL="http://localhost:$FASTAPI_PORT"

# Function to handle shutdown
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    if [ ! -z "$FASTAPI_PID" ]; then
        kill $FASTAPI_PID 2>/dev/null || true
    fi
    if [ ! -z "$STREAMLIT_PID" ]; then
        kill $STREAMLIT_PID 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    echo "✅ Services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT EXIT

# Start FastAPI in background (non-blocking - don't wait for it)
echo "⏳ Starting FastAPI backend in background..."
cd /app

# Start FastAPI - allow it to fail without stopping the script
# Don't wait for it - Streamlit needs to start immediately for Railway
python -m uvicorn api:app \
    --host 0.0.0.0 \
    --port $FASTAPI_PORT \
    --log-level info \
    --workers 1 \
    --no-reload \
    --timeout-keep-alive 30 \
    --timeout-graceful-shutdown 30 > /tmp/fastapi.log 2>&1 &
FASTAPI_PID=$!

echo "⏳ FastAPI starting in background (PID: $FASTAPI_PID)..."
echo "   (Will be available once model loading completes - check /tmp/fastapi.log)"
echo "   Streamlit will start immediately (can work in fallback mode)"

# Start Streamlit immediately (this will be the main process)
# Railway needs a responding service quickly - don't wait for FastAPI
echo ""
echo "⏳ Starting Streamlit frontend..."
echo "============================================================"
echo "📍 Starting Streamlit on port $EXTERNAL_PORT"
echo "   FastAPI is starting in background (will be available once ready)"
echo "   Streamlit will work in fallback mode until FastAPI is ready"
echo "============================================================"
echo ""
echo "🚀 Streamlit starting now..."
echo ""

# Set Streamlit environment variables
export STREAMLIT_SERVER_PORT=$EXTERNAL_PORT
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Run Streamlit in foreground (Railway needs a foreground process)
# Use python -m streamlit to avoid PATH issues
# Don't exit on error - let Streamlit handle it
exec python -m streamlit run app.py \
    --server.port=$EXTERNAL_PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
