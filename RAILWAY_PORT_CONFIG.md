# Railway Port Configuration Guide

## Quick Answer

**If Railway is asking for the port, you have two options:**

### Option 1: Let Railway Auto-Detect (Recommended)
- **Don't enter anything** - Railway automatically sets the `PORT` environment variable
- The startup script (`start_services.sh`) automatically uses this port
- Streamlit will listen on whatever port Railway assigns

### Option 2: Set Port Manually
If Railway requires you to enter a port:

1. **Go to Railway Dashboard** → Your Service → **Settings** tab
2. Look for **"Port"** or **"Listening Port"** field
3. **Enter:** `8501` (or leave blank to use Railway's auto-assigned port)
4. **OR** add environment variable:
   - Name: `PORT`
   - Value: `8501` (or leave Railway to auto-assign)

## How It Works

The startup script automatically handles ports:

```bash
# Gets port from Railway's PORT environment variable
EXTERNAL_PORT=${PORT:-8501}  # Uses PORT if set, otherwise 8501
FASTAPI_PORT=${FASTAPI_PORT:-8000}  # FastAPI on internal port 8000
```

- **Streamlit** (main service): Uses `PORT` env var (Railway sets this)
- **FastAPI** (internal): Uses port 8000 (not exposed externally)

## Recommended Configuration

**In Railway Settings:**

1. **Port Field:** Leave blank OR enter `8501`
2. **Environment Variables:**
   - `PORT` - Railway sets this automatically (don't override)
   - `FASTAPI_PORT=8000` - Internal FastAPI port
   - `GROQ_API_KEY` - Your Groq API key
   - `FASTAPI_URL=http://localhost:8000`

## Verification

After setting, check Railway logs. You should see:
```
📍 Streamlit will run on port: [PORT_NUMBER] (external)
📍 FastAPI will run on port: 8000 (internal)
```

The PORT_NUMBER will be whatever Railway assigned (usually a random high port like 30000+).

## Troubleshooting

**If Railway keeps asking for port:**
1. Go to **Settings** → **Networking**
2. Check if **"Public Port"** is set
3. If not, Railway should auto-assign - just proceed with deployment
4. The app will work regardless of the port number Railway assigns

**Important:** Railway automatically routes traffic to your app, so the actual port number doesn't matter - Railway handles the routing!
