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

# Start FastAPI in background (don't fail if it doesn't start immediately)
echo "⏳ Starting FastAPI backend..."
cd /app

# Start FastAPI - allow it to fail without stopping the script
python -m uvicorn api:app \
    --host 0.0.0.0 \
    --port $FASTAPI_PORT \
    --log-level info \
    --workers 1 \
    --no-reload \
    --timeout-keep-alive 30 \
    --timeout-graceful-shutdown 30 > /tmp/fastapi.log 2>&1 &
FASTAPI_PID=$!

echo "⏳ FastAPI starting (PID: $FASTAPI_PID)..."
echo "   Check /tmp/fastapi.log for FastAPI output"

# Wait for FastAPI to start (with retries)
echo "⏳ Waiting for FastAPI to initialize..."
MAX_WAIT=60
WAIT_COUNT=0
FASTAPI_READY=0

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -f -s http://localhost:$FASTAPI_PORT/health > /dev/null 2>&1; then
        echo "✅ FastAPI is ready!"
        FASTAPI_READY=1
        break
    fi
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
    echo "   Still waiting... ($WAIT_COUNT/$MAX_WAIT seconds)"
done

if [ $FASTAPI_READY -eq 0 ]; then
    echo "⚠️  FastAPI did not start within $MAX_WAIT seconds"
    echo "   Streamlit will start anyway (fallback mode)"
    echo "   FastAPI logs:"
    tail -20 /tmp/fastapi.log 2>/dev/null || echo "   (No logs available)"
fi

# Start Streamlit in foreground (this will be the main process)
echo ""
echo "⏳ Starting Streamlit frontend..."
echo "============================================================"
if [ $FASTAPI_READY -eq 1 ]; then
    echo "✅ FastAPI is running on port $FASTAPI_PORT"
else
    echo "⚠️  FastAPI not ready - Streamlit will use fallback mode"
fi
echo "📍 Starting Streamlit on port $EXTERNAL_PORT"
echo "============================================================"
echo ""
echo "⏳ Application is loading..."
echo "   (This may take 1-2 minutes for model loading on first start)"
echo ""

# Set Streamlit environment variables
export STREAMLIT_SERVER_PORT=$EXTERNAL_PORT
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Run Streamlit in foreground (Railway needs a foreground process)
# Don't exit on error - let Streamlit handle it
exec streamlit run app.py \
    --server.port=$EXTERNAL_PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
