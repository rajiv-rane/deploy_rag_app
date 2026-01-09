# 🔧 Troubleshooting Deployment Issues

## Common Issues and Solutions

### Issue 1: 502 Bad Gateway Error

**Symptoms:**
- Connection error 502
- "Application failed to respond"

**Causes:**
1. Application is still starting (model loading takes 2-3 minutes)
2. Streamlit not binding to correct port
3. Health check failing

**Solutions:**

1. **Wait for startup** (2-3 minutes on first deployment)
   - Check Railway logs to see "Starting Streamlit frontend..."
   - Wait for "Application startup complete"

2. **Check Railway Logs:**
   - Go to Railway dashboard → Your service → **Logs** tab
   - Look for errors or startup messages
   - Verify Streamlit is starting on the correct port

3. **Verify Environment Variables:**
   - `GROQ_API_KEY` - Must be set
   - `FASTAPI_URL` - Should be `http://localhost:8000`
   - `PORT` - Railway sets this automatically

4. **Check Service Status:**
   - In Railway dashboard, verify service shows "Active"
   - If it shows "Failed", check logs for errors

---

### Issue 2: FastAPI Backend Not Available

**Symptoms:**
- Warning: "FastAPI backend not available. Using fallback mode."
- App works but slower

**This is OK!** The app works in fallback mode. But to enable FastAPI:

**Solutions:**

1. **Check Railway Logs:**
   - Look for FastAPI startup messages
   - Check if there are any errors in `/tmp/fastapi.log`

2. **Verify FastAPI is starting:**
   - In logs, you should see: "⏳ Starting FastAPI backend..."
   - Then: "✅ FastAPI is ready!" (after 60 seconds max)

3. **If FastAPI doesn't start:**
   - Check if models are loading (takes time)
   - Verify MongoDB connection is working
   - Check GROQ_API_KEY is set correctly

4. **Fallback mode is fine:**
   - The app will work, just slower
   - All features are available
   - FastAPI is optional for better performance

---

### Issue 3: AutoGen Not Available

**Symptoms:**
- Warning: "AutoGen not available. Some features may be limited."

**This is expected!** AutoGen is optional and not needed for core functionality.

**Solutions:**
- This warning is safe to ignore
- All core features work without AutoGen
- The app uses direct Groq API calls instead

---

### Issue 4: TypeError: Failed to fetch dynamically imported module

**Symptoms:**
- Error in sidebar about module import
- App still works but shows errors

**Causes:**
- Streamlit trying to load modules that don't exist
- CORS or network issues

**Solutions:**

1. **Refresh the page** - Often fixes temporary issues
2. **Clear browser cache** - Ctrl+Shift+R (or Cmd+Shift+R on Mac)
3. **Check Railway logs** - Verify Streamlit is running correctly
4. **This is usually harmless** - App functionality is not affected

---

### Issue 5: Slow Loading / Timeouts

**Symptoms:**
- App takes 2-3 minutes to load
- Requests timeout

**This is normal on first startup!**

**Why:**
- Bio ClinicalBERT model downloads (~400MB)
- Models load into memory
- Vector database initializes

**Solutions:**

1. **Wait patiently** - First load takes 2-3 minutes
2. **Check logs** - You'll see "Downloading model..." messages
3. **Subsequent loads** - Much faster (models cached)
4. **For exhibition** - Start the app 10 minutes before to ensure it's ready

---

## 🔍 How to Check Railway Logs

1. Go to Railway dashboard
2. Click on your service
3. Click **"Logs"** tab (left sidebar)
4. Look for:
   - ✅ "Starting Streamlit frontend..."
   - ✅ "FastAPI is ready!"
   - ❌ Any error messages

---

## ✅ Quick Health Check

After deployment, verify:

1. **Service Status:** Shows "Active" in Railway
2. **Logs:** No critical errors
3. **URL:** App loads (even if slow on first load)
4. **Features:** Can search for patients, use AI assistant

---

## 🚨 If Nothing Works

1. **Check Environment Variables:**
   - Railway dashboard → Service → Variables tab
   - Verify `GROQ_API_KEY` is set correctly

2. **Redeploy:**
   - Railway dashboard → Service → Settings
   - Click "Redeploy" or create new deployment

3. **Check Resource Limits:**
   - Free tier has limits
   - Consider upgrading if hitting limits

4. **Contact Support:**
   - Railway has good support
   - Check their documentation

---

## 📞 Emergency Backup Plan

If deployment fails completely:

1. **Use ngrok for local deployment:**
   ```bash
   # Terminal 1
   cd ingestion-phase
   python start_api.py
   
   # Terminal 2
   cd ingestion-phase
   streamlit run app.py
   
   # Terminal 3
   ngrok http 8501
   ```

2. **Share ngrok URL** for exhibition

---

**Most issues resolve after waiting 2-3 minutes for initial startup!**
