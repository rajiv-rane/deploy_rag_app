"""
FastAPI Backend for Medical Discharge Summary Assistant
Provides async endpoints for AI agent and RAG operations
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
import asyncio
import httpx
import torch
import chromadb
from motor.motor_asyncio import AsyncIOMotorClient
import hashlib
import json
import time
import math
from contextlib import asynccontextmanager
from bson import ObjectId
from bson.errors import InvalidId
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Get the directory where this file is located
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Try to import transformers, but make it optional
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModel = None

# Configuration
# Use environment variable if available, otherwise fallback to default
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://ishaanroopesh0102:6eShFuC0pNnFFNGm@cluster0.biujjg4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
CHROMA_PATH = os.getenv("CHROMA_PATH", "vector_db/chroma")

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

# Legacy Ollama config (deprecated)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"

# Global state
embedding_cache = {}
tokenizer = None
model = None
chroma_client = None
chroma_collection = None
mongo_client = None
mongo_db = None
patients_collection = None
http_client = None

# Request models
class ChatRequest(BaseModel):
    message: str
    patient_data: Optional[Dict[str, Any]] = None

class SummaryRequest(BaseModel):
    patient_data: str
    template_outline: Optional[List[str]] = None

class SearchRequest(BaseModel):
    query_text: str
    n_results: int = 3

class PatientRequest(BaseModel):
    unit_no: str

# Initialize models and connections
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global tokenizer, model, chroma_client, chroma_collection
    global mongo_client, mongo_db, patients_collection, http_client
    
    # Load models (with timeout to prevent blocking startup)
    if not TRANSFORMERS_AVAILABLE:
        print("⚠️ Transformers library not available. Some features may be limited.")
        tokenizer = None
        model = None
    else:
        try:
            print("Loading Bio ClinicalBERT model...")
            print("   (This may take 1-2 minutes on first run - downloading ~400MB model)")
            # Set cache directory to avoid permission issues
            # Use HF_HOME (new) instead of TRANSFORMERS_CACHE (deprecated)
            os.environ["HF_HOME"] = os.getenv("HF_HOME", "/app/.cache/huggingface")
            os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "/app/.cache/transformers")  # For backward compatibility
            tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", cache_dir="/app/.cache/transformers")
            model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", cache_dir="/app/.cache/transformers")
            model.eval()
            if torch.cuda.is_available():
                model.to("cuda")
            print("✅ Bio ClinicalBERT model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Error loading transformers model: {str(e)}")
            print(f"   Full error: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            tokenizer = None
            model = None
            print("   FastAPI will continue without embedding model (some features disabled)")
    
    # Connect to ChromaDB
    print("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = chroma_client.get_or_create_collection("patient_embeddings")
    
    # Connect to MongoDB
    print("Connecting to MongoDB...")
    mongo_client = AsyncIOMotorClient(MONGO_URI)
    mongo_db = mongo_client["hospital_db"]
    patients_collection = mongo_db["test_patients"]
    
    # Create async HTTP client (no timeout for long-running LLM requests)
    http_client = httpx.AsyncClient(
        timeout=None,  # No timeout - allow long-running requests
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
    )
    
    print("✅ FastAPI backend initialized successfully!")
    
    try:
        yield
    except asyncio.CancelledError:
        # Handle cancellation during reload/shutdown gracefully
        # Don't re-raise, allow cleanup to proceed
        pass
    finally:
        # Cleanup - handle cancellation gracefully during shutdown
        try:
            if http_client:
                try:
                    await http_client.aclose()
                except (asyncio.CancelledError, RuntimeError):
                    # Ignore cancellation and runtime errors during shutdown
                    pass
        except Exception as e:
            # Log but don't raise during shutdown
            pass
        
        try:
            if mongo_client:
                mongo_client.close()
        except Exception:
            # Ignore errors during shutdown
            pass
        
        try:
            print("✅ FastAPI backend shutdown complete")
        except:
            # Even print might fail during shutdown, ignore
            pass

# Create FastAPI app
app = FastAPI(
    title="Medical Discharge Summary API",
    description="Fast async API for medical RAG and AI agent operations",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions
def get_text_hash(text: str) -> str:
    """Generate hash for text to use as cache key"""
    return hashlib.md5(text.encode()).hexdigest()

async def embed_text_async(text: str) -> List[float]:
    """Generate embedding for text using Bio ClinicalBERT with caching"""
    # Check cache first
    text_hash = get_text_hash(text)
    if text_hash in embedding_cache:
        return embedding_cache[text_hash]
    
    # Generate embedding (run in thread pool to avoid blocking)
    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(
        None,
        lambda: _generate_embedding(text)
    )
    
    # Cache the embedding
    embedding_cache[text_hash] = embedding
    return embedding

def _generate_embedding(text: str) -> List[float]:
    """Synchronous embedding generation"""
    if not TRANSFORMERS_AVAILABLE or tokenizer is None or model is None:
        raise RuntimeError("Transformers model not available. Please install transformers library.")
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        emb = cls_embedding.squeeze(0)
        if emb.is_cuda:
            emb = emb.to("cpu")
        return emb.tolist()

async def check_groq_available() -> Tuple[bool, str]:
    """Check if Groq API is accessible. Returns (is_available, status_message)"""
    if not GROQ_API_KEY:
        return (False, "GROQ_API_KEY environment variable is not set. Please set it in your .env file or environment.")
    
    try:
        # Simple test request to check API key validity
        test_payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 5
        }
        response = await http_client.post(
            GROQ_BASE_URL + "/chat/completions",
            json=test_payload,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=10.0
        )
        if response.status_code == 200:
            return (True, f"Groq API is accessible with model {GROQ_MODEL}")
        elif response.status_code == 401:
            return (False, "Invalid Groq API key. Please check your GROQ_API_KEY.")
        elif response.status_code == 404:
            return (False, f"Model {GROQ_MODEL} not found. Please check the model name.")
        else:
            return (False, f"Groq API returned status {response.status_code}")
    except httpx.TimeoutException:
        return (False, "Groq API request timed out. Please check your internet connection.")
    except httpx.RequestError as e:
        return (False, f"Cannot connect to Groq API: {str(e)}")
    except Exception as e:
        return (False, f"Error checking Groq API: {str(e)}")

def clean_markdown_formatting(text: str) -> str:
    """Remove markdown formatting like asterisks, bold markers, etc."""
    import re
    # Remove bold/italic markers (**text**, *text*)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove **bold**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Remove *italic*
    # Remove standalone asterisks used as bullets
    text = re.sub(r'^\s*\*\s+', '', text, flags=re.MULTILINE)
    # Remove markdown headers (# Header)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    return text

async def call_groq_async(messages: List[Dict], options: Dict = None, max_retries: int = 3) -> str:
    """Async call to Groq API with retry logic"""
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY environment variable is not set. Please set it in your .env file or environment variables."
        )
    
    # Map options to Groq API format
    groq_options = {
        "temperature": options.get("temperature", 0.3) if options else 0.3,
        "top_p": options.get("top_p", 0.85) if options else 0.85,
        "max_tokens": options.get("max_tokens", 4000) if options else 4000,
    }
    
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        **groq_options
    }
    
    # Retry logic for production reliability
    last_error = None
    for attempt in range(max_retries):
        try:
            response = await http_client.post(
                GROQ_BASE_URL + "/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                timeout=120.0  # 2 minute timeout for long responses
            )
            
            # Check for server errors
            if response.status_code == 401:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid Groq API key. Please check your GROQ_API_KEY environment variable."
                )
            elif response.status_code == 429:
                # Rate limit - wait longer and retry with exponential backoff
                if attempt < max_retries - 1:
                    wait_time = min(2 ** (attempt + 2), 30)  # Exponential backoff: 4s, 8s, 16s (max 30s)
                    await asyncio.sleep(wait_time)
                    continue
                # Check for retry-after header if available
                retry_after = response.headers.get("retry-after", None)
                error_msg = "Groq API rate limit exceeded. Please wait a few moments before trying again."
                if retry_after:
                    error_msg += f" Retry after {retry_after} seconds."
                raise HTTPException(
                    status_code=429,
                    detail=error_msg
                )
            elif response.status_code == 500:
                # Server error - retry
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                raise HTTPException(
                    status_code=503,
                    detail="Groq API server error. Please try again later."
                )
            
            response.raise_for_status()
            data = response.json()
            
            # Extract response content
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content", "")
                # Clean markdown formatting (remove asterisks, etc.)
                cleaned_content = clean_markdown_formatting(content)
                return cleaned_content.strip() if cleaned_content else "No response generated."
            else:
                raise HTTPException(
                    status_code=502,
                    detail="Invalid response format from Groq API."
                )
                
        except httpx.TimeoutException:
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            raise HTTPException(
                status_code=504,
                detail="Request to Groq API timed out. The model may be processing a large request."
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model {GROQ_MODEL} not found. Please check the model name in configuration."
                )
            last_error = e
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
    
    # If all retries failed
    error_msg = str(last_error) if last_error else "Unknown error"
    # Check if the error is related to rate limiting
    if "429" in error_msg or "rate limit" in error_msg.lower():
        raise HTTPException(
            status_code=429,
            detail="Groq API rate limit exceeded. Please wait a few moments before trying again. Consider upgrading your Groq API plan for higher rate limits."
        )
    raise HTTPException(
        status_code=502,
        detail=f"Failed to connect to Groq API after {max_retries} attempts: {error_msg}"
    )

# Legacy Ollama function (kept for backward compatibility)
async def check_ollama_available() -> Tuple[bool, str]:
    """Check if Ollama is running and accessible. Returns (is_available, status_message)"""
    try:
        response = await http_client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        if response.status_code == 200:
            # Check if llama3 model is available
            try:
                data = response.json()
                models = data.get("models", [])
                model_names = [m.get("name", "") for m in models]
                has_llama3 = any("llama3" in name for name in model_names)
                if has_llama3:
                    return (True, "Ollama is running with llama3 model")
                else:
                    return (False, "Ollama is running but llama3 model not found. Run: ollama pull llama3")
            except:
                return (True, "Ollama is running")
        return (False, f"Ollama returned status {response.status_code}")
    except httpx.ConnectError:
        # Cannot connect - Ollama is not running
        return (False, "Cannot connect to Ollama. Is it running? (run 'ollama serve')")
    except httpx.HTTPStatusError as e:
        # If we get a response, Ollama is running (even if endpoint returns error)
        return (e.response.status_code != 404, f"Ollama responded with status {e.response.status_code}")
    except Exception as e:
        return (False, f"Error checking Ollama: {str(e)}")

async def call_ollama_async(messages: List[Dict], options: Dict = None) -> str:
    """Async call to Ollama API"""
    # Check if Ollama is available first
    is_available, status_msg = await check_ollama_available()
    if not is_available:
        raise HTTPException(
            status_code=503, 
            detail=f"Ollama service is not available. {status_msg}\n\n"
                   f"If you recently changed Ollama's model directory:\n"
                   f"1. Restart Ollama: Stop 'ollama serve' and start it again\n"
                   f"2. Re-download models: Run 'ollama pull llama3'\n"
                   f"3. Verify: Run 'ollama list' to see available models"
        )
    
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True
    }
    if options:
        payload["options"] = options
    
    try:
        response = await http_client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=None  # No timeout - allow long-running LLM generation
        )
        
        # Check for server errors
        if response.status_code == 500:
            raise HTTPException(
                status_code=503,
                detail="Ollama server returned an error. Please check if the LLaMA 3 model is installed (run 'ollama pull llama3')."
            )
        
        response.raise_for_status()
        
        full_response = ""
        async for line in response.aiter_lines():
            if not line:
                continue
            try:
                json_data = json.loads(line)
                if 'message' in json_data and 'content' in json_data['message']:
                    content = json_data['message']['content']
                    if content:
                        full_response += content
                if json_data.get('done', False):
                    break
            except json.JSONDecodeError:
                continue
        
        return full_response.strip()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to Ollama timed out. The model may be too large or the system is overloaded.")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(
                status_code=503,
                detail=f"Ollama API endpoint not found (404). This usually means:\n"
                       f"1. Ollama is not running - Start it with 'ollama serve' in a terminal\n"
                       f"2. Ollama model directory was changed - Restart Ollama after changing model directory\n"
                       f"3. Model not in new directory - Run 'ollama pull llama3' to download to new location\n"
                       f"4. Ollama is running on a different port - Check if it's accessible at {OLLAMA_BASE_URL}\n"
                       f"5. Ollama version is outdated - Update with 'ollama --version' and reinstall if needed\n\n"
                       f"To verify Ollama is running, open: http://localhost:11434/api/tags\n"
                       f"To check available models, run: ollama list"
            )
        elif e.response.status_code == 500:
            raise HTTPException(
                status_code=503,
                detail="Ollama server error. Please check: 1) Ollama is running ('ollama serve'), 2) LLaMA 3 model is installed ('ollama pull llama3'), 3) System has enough resources."
            )
        raise HTTPException(
            status_code=502, 
            detail=f"Error connecting to Ollama (HTTP {e.response.status_code}): {str(e)}\n"
                   f"Please ensure Ollama is running at {OLLAMA_BASE_URL}"
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Please ensure Ollama is running (run 'ollama serve' in a terminal)."
        )

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Medical Discharge Summary API",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Simple root endpoint - responds immediately even during startup"""
    return {
        "status": "online",
        "service": "Medical Discharge Summary API",
        "version": "1.0.0",
        "message": "API is starting up. Use /health for detailed status."
    }

@app.get("/health")
async def health():
    """Detailed health check - may take time if models are loading"""
    try:
        groq_status = "unknown"
        groq_message = ""
        try:
            groq_available, status_msg = await check_groq_available()
            groq_status = "connected" if groq_available else "disconnected"
            groq_message = status_msg
        except Exception as e:
            groq_status = "error checking"
            groq_message = str(e)
        
        return {
            "status": "healthy",
            "mongodb": "connected" if mongo_client else "disconnected",
            "chromadb": "connected" if chroma_client else "disconnected",
            "model": "loaded" if model else "not loaded",
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "groq": groq_status,
            "groq_message": groq_message,
            "groq_model": GROQ_MODEL,
            "groq_url": GROQ_BASE_URL,
            "api_key_set": bool(GROQ_API_KEY),
        "cache_size": len(embedding_cache)
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat with AI agent - optimized for speed"""
    try:
        # Check if user is asking for discharge summary
        if "discharge summary" in request.message.lower() or "generate summary" in request.message.lower():
            if request.patient_data:
                patient_text = format_patient_fields(request.patient_data)
                summary = await generate_summary_async(patient_text)
                return {"response": summary}
            else:
                return {"response": "❌ Please select a patient first to generate a discharge summary."}
        
        # Add patient context with actual patient data if available
        patient_context = ""
        if request.patient_data:
            # Format patient data for context
            patient_text = format_patient_fields(request.patient_data)
            patient_name = request.patient_data.get('name', 'Unknown')
            unit_no = request.patient_data.get('unit no', 'N/A')
            patient_context = f"""

CURRENT PATIENT INFORMATION:
{patient_text}

Note: The person you are talking to is a medical professional (doctor or nurse) asking questions about this patient. Answer their questions based on the patient information provided above."""
        
        system_prompt = """You are a medical AI assistant designed to help doctors and nurses with clinical documentation and medical questions. 
The person you are talking to is a medical professional (doctor or nurse), NOT a patient. 
When patient information is provided, use it to answer questions about that specific patient.
IMPORTANT: Only use the patient information provided in the CURRENT PATIENT INFORMATION section below. Do not reference any previous patients or conversations.
Provide accurate, helpful responses based on the patient data. Be concise but thorough."""
        
        user_message = request.message
        if patient_context:
            # Add explicit instruction to use only the current patient
            user_message = f"{request.message}\n\n{patient_context}\n\nRemember: Only answer questions about the patient information provided above. Do not reference any other patients."
        else:
            # If no patient data, remind that patient needs to be selected
            user_message = f"{request.message}\n\nNote: No patient is currently selected. Please select a patient first to get patient-specific information."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        options = {
            "temperature": 0.4,
            "top_p": 0.85,
            "max_tokens": 500,  # Increased to allow proper responses
            "num_predict": 500
        }
        
        response = await call_groq_async(messages, options)
        return {"response": response if response else "I'm here to help with medical questions. How can I assist you?"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

@app.post("/api/generate-summary")
async def generate_summary(request: SummaryRequest):
    """Generate discharge summary - async optimized"""
    try:
        if request.template_outline:
            summary = await generate_summary_with_template_async(request.patient_data, request.template_outline)
        else:
            summary = await generate_summary_async(request.patient_data)
        
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

async def generate_summary_async(patient_data: str) -> str:
    """Generate discharge summary using Groq API"""
    system_prompt = """You are an expert medical AI assistant that generates structured, clinically accurate discharge summaries.
Base your summary entirely on the INPUT PATIENT DATA provided.
The discharge summary MUST include: Name, Unit No, Date Of Birth, Sex, Admission/Discharge Dates, Attending, Chief Complaint, Procedure, History, Physical Exam (on Admission), Pertinent Results, Brief Hospital Course, Medications on Admission, Discharge Medications, Discharge Instructions, Discharge Disposition, Discharge Diagnosis, Discharge Condition, Follow-up.

FORMATTING REQUIREMENTS:
- Use clear section headings on separate lines (e.g., "Name:", "Unit No:", "Admission Date:", etc.)
- Put each section on its own line with proper spacing
- Use line breaks between major sections
- For Name, Unit No, Date of Birth, and Sex, put them on separate lines at the top
- For lists (medications, diagnoses), put each item on a new line or separate with commas clearly
- If information is missing, state "[Information not available]"
- Use concise, professional medical language. Be brief and factual.
- Do NOT use markdown formatting (no asterisks, no bold, no headers). Use plain text with line breaks for structure.

EXAMPLE FORMAT:
Name: [Name]
Unit No: [Unit No]
Date of Birth: [DOB]
Sex: [Sex]

Admission Date: [Date]
Discharge Date: [Date]
Attending: [Name]

Chief Complaint: [Complaint]

Procedure: [Procedure details]

History Of Present Illness: [History]

Past Medical History: [History]

Physical Exam: [Exam details]

[Continue with other sections, each on separate lines]"""

    user_prompt = f"""Generate a discharge summary for this patient:
{patient_data}

Extract Name, Unit No, Date of Birth, and Sex exactly as provided."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    options = {
        "temperature": 0.3,
        "top_p": 0.85,
        "max_tokens": 4000,
        "num_predict": 4000
    }
    
    return await call_groq_async(messages, options)

async def generate_summary_with_template_async(patient_data: str, outline_sections: List[str]) -> str:
    """Generate discharge summary following template outline"""
    outline_bullets = "\n".join([f"- {s}" for s in outline_sections])
    system_prompt = f"""You are an expert medical AI assistant that generates a clinically accurate discharge summary.
Follow the section order EXACTLY as specified by the provided outline. Do not add extra sections; if information is missing, write "[Information not available]".

REQUIRED SECTION ORDER (USE EXACT TITLES):
{outline_bullets}

FORMATTING REQUIREMENTS:
- Use clear section headings on separate lines (e.g., "Name:", "Unit No:", etc.)
- Put each section on its own line with proper spacing
- Use line breaks between major sections
- For lists (medications, diagnoses), put each item on a new line or separate with commas clearly
- Use concise, professional medical language
- Base content solely on the input patient data
- Preserve patient identifiers verbatim if present
- Be brief and factual
- Do NOT use markdown formatting (no asterisks, no bold, no headers). Use plain text with line breaks for structure."""

    user_prompt = f"""Generate a discharge summary STRICTLY following the section list above, based only on this data:\n\n{patient_data}\n\nReturn plain text with the exact section headings in order."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    options = {
        "temperature": 0.3,
        "top_p": 0.85,
        "max_tokens": 4000,
        "num_predict": 4000
    }
    
    return await call_groq_async(messages, options)

@app.post("/api/search-similar")
async def search_similar(request: SearchRequest):
    """Search for similar cases using RAG - excludes MongoDB patients"""
    try:
        # Check if ChromaDB is initialized
        if chroma_collection is None:
            raise HTTPException(
                status_code=503,
                detail="ChromaDB collection not initialized. Please check if the vector database is set up correctly."
            )
        
        # Check if transformers model is available
        if not TRANSFORMERS_AVAILABLE or tokenizer is None or model is None:
            raise HTTPException(
                status_code=503,
                detail="Embedding model not available. Please ensure transformers library is installed and models are loaded."
            )
        
        # Get list of MongoDB patient unit numbers to exclude
        mongo_unit_nos = set()
        try:
            if mongo_client and patients_collection:
                mongo_patients = await patients_collection.find({}, {"unit no": 1, "_id": 0}).to_list(length=None)
                mongo_unit_nos = {str(patient.get("unit no", "")) for patient in mongo_patients if patient.get("unit no")}
        except Exception as e:
            print(f"Warning: Could not fetch MongoDB patients for filtering: {str(e)}")
        
        # Generate embedding
        query_embedding = await embed_text_async(request.query_text)
        
        # Search ChromaDB with more results to account for filtering
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=request.n_results * 3,  # Get more results to filter
                include=["documents", "metadatas", "distances"]
            )
        )
        
        # Check if results are valid
        if not results or "documents" not in results or len(results["documents"]) == 0:
            return {"similar_cases": []}
        
        if len(results["documents"][0]) == 0:
            return {"similar_cases": []}
        
        similar_cases = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0] if "metadatas" in results and len(results["metadatas"]) > 0 else [{}] * len(documents)
        distances = results["distances"][0] if "distances" in results and len(results["distances"]) > 0 else [0.0] * len(documents)
        
        for i in range(len(documents)):
            metadata = metadatas[i] if i < len(metadatas) else {}
            unit_no = str(metadata.get('unit_no') or metadata.get('unit no', ''))
            
            # Skip if this patient exists in MongoDB (only show ChromaDB/preprocessed patients)
            if unit_no and unit_no in mongo_unit_nos:
                continue
            
            similar_cases.append({
                "document": documents[i],
                "metadata": metadata,
                "similarity": 1 - distances[i] if i < len(distances) else 0.0
            })
            
            # Stop when we have enough results
            if len(similar_cases) >= request.n_results:
                break
        
        return {"similar_cases": similar_cases}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error searching similar cases: {str(e)}\n\nTraceback:\n{error_trace}"
        )

@app.post("/api/patient")
async def get_patient(request: PatientRequest):
    """Get patient by unit number"""
    try:
        # Clean and validate unit_no
        unit_no_str = str(request.unit_no).strip()
        if not unit_no_str or unit_no_str == 'Unknown':
            raise HTTPException(status_code=400, detail="Invalid unit number: empty or 'Unknown'")
        
        # Try to convert unit_no to int
        try:
            unit_no_int = int(unit_no_str)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid unit number format: '{unit_no_str}'. Expected a numeric value."
            )
        
        # Query MongoDB - try both int and string formats
        patient = await patients_collection.find_one({"unit no": unit_no_int})
        if not patient:
            # Try as string as fallback
            patient = await patients_collection.find_one({"unit no": unit_no_str})
        
        if not patient:
            raise HTTPException(
                status_code=404, 
                detail=f"Patient with unit number '{unit_no_str}' not found in database"
            )
        
        # Clean the patient data (handle NaN, ObjectId, etc.)
        cleaned_patient = clean_patient_data(patient)
        
        return {"patient": cleaned_patient}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving patient: {str(e)}")

@app.post("/api/embed")
async def embed_text_endpoint(request: Dict[str, str]):
    """Generate embedding for text using Bio ClinicalBERT"""
    if "text" not in request:
        raise HTTPException(status_code=400, detail="Missing 'text' field in request")
    
    text = request["text"]
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Check if model is available
    if not TRANSFORMERS_AVAILABLE or tokenizer is None or model is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding model is still loading. Please wait 1-2 minutes and try again. The model takes 2-3 minutes to load on first startup."
        )
    
    try:
        embedding = await embed_text_async(text)
        return {"embedding": embedding}
    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Embedding model not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating embedding: {str(e)}"
        )

@app.get("/api/patients")
async def get_all_patients():
    """Get list of all patients with name and unit number"""
    try:
        # Query MongoDB for all patients, only get name and unit no
        cursor = patients_collection.find(
            {},
            {"name": 1, "unit no": 1, "_id": 0}  # Only return name and unit no
        )
        patients = await cursor.to_list(length=None)  # Get all patients
        
        # Clean and format patient data
        patient_list = []
        for patient in patients:
            cleaned = clean_patient_data(patient)
            name = cleaned.get("name", "Unknown")
            unit_no = cleaned.get("unit no", "N/A")
            if name and name != "Unknown" and unit_no and unit_no != "N/A":
                patient_list.append({
                    "name": str(name),
                    "unit_no": str(unit_no),
                    "display": f"{name} (Unit {unit_no})"
                })
        
        # Sort by name
        patient_list.sort(key=lambda x: x["name"])
        
        return {"patients": patient_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving patients: {str(e)}")

def clean_patient_data(patient: Dict) -> Dict:
    """Clean patient data for JSON serialization - handle NaN, ObjectId, etc."""
    cleaned = {}
    for key, value in patient.items():
        # Handle NaN values
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            cleaned[key] = None
        # Handle ObjectId
        elif isinstance(value, ObjectId):
            cleaned[key] = str(value)
        # Handle nested dictionaries
        elif isinstance(value, dict):
            cleaned[key] = clean_patient_data(value)
        # Handle lists
        elif isinstance(value, list):
            cleaned[key] = [
                clean_patient_data(item) if isinstance(item, dict) else 
                (None if isinstance(item, float) and (math.isnan(item) or math.isinf(item)) else item)
                for item in value
            ]
        # Handle other types
        else:
            cleaned[key] = value
    return cleaned

def format_patient_fields(record: Dict) -> str:
    """Format patient record fields for embedding"""
    fields = [
        "name", "unit no", "admission date", "date of birth", "sex", "service",
        "allergies", "attending", "chief complaint", "major surgical or invasive procedure",
        "history of present illness", "past medical history", "social history",
        "family history", "physical exam", "pertinent results", "medications on admission",
        "brief hospital course", "discharge medications", "discharge diagnosis",
        "discharge condition", "discharge instructions", "follow-up", "discharge disposition"
    ]
    parts = [f"{field.title()}: {record.get(field, '')}" for field in fields if record.get(field)]
    return " ".join(parts)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


