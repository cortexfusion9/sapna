import os
import json
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from vector_service import semantic_secure_search, ingest_project_data

# Load local environment variables from .env file
load_dotenv()

app = FastAPI(
    title="General Contractor Secure SOP Agent",
    description="Full-stack Secure RAG Agent with ChromaDB Vector Search + Groq LLM + RLS.",
    version="3.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic request schemas
class LoginPayload(BaseModel):
    username: str

class ChatRequest(BaseModel):
    message: str


def get_groq_client():
    """Initializes Groq client securely from env variable."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        return None
    try:
        return Groq(api_key=api_key.strip())
    except Exception as e:
        print(f"[ERROR] Groq client initialization failed: {str(e)}")
        return None


# On startup: ingest Excel data into local ChromaDB vector store
@app.on_event("startup")
def startup_event():
    print("[STARTUP] Ingesting project data into ChromaDB Vector Store...")
    ingest_project_data()
    print("[STARTUP] Vector store ready.")


# ── Login endpoint (no password – match username against user_id) ──
@app.post("/api/auth/login")
def secure_login(payload: LoginPayload):
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    df = pd.read_excel(os.path.join(BASE_DIR, "users_gate_permissions.xlsx"), dtype=str).fillna("")
    df.columns = [c.strip().lower() for c in df.columns]

    match = df[df["user_id"] == payload.username.strip()]
    if match.empty:
        raise HTTPException(status_code=401, detail="Invalid corporate credentials")

    row = match.iloc[0]
    return {
        "status": "authenticated",
        "token": row.get("user_id"),
        "name": row.get("user_name", payload.username),
        "role": row.get("user_role", "Unknown"),
    }


# Serve the frontend HTML at root
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)


# Secure Chat Endpoint wired to ChromaDB + Groq
@app.post("/api/chat")
def secure_chat_endpoint(
    req: ChatRequest,
    x_acting_as: str | None = Header(default=None, alias="X-Acting-As")
):
    """
    Full E2E secure endpoint:
    1. Validates X-User-Id header.
    2. Runs semantic search via ChromaDB with strict RLS metadata filtering.
    3. Feeds retrieved context to Groq (Llama 3.3) for humanized response.
    """
    # 1. Validate security header
    if not x_acting_as or not x_acting_as.strip():
        raise HTTPException(
            status_code=401,
            detail="Missing required 'X-Acting-As' header."
        )

    user_id_val = x_acting_as.strip()

    # 2. Semantic search with RLS enforcement (ChromaDB)
    vector_context = semantic_secure_search(user_id=user_id_val, user_query=req.message)

    # Handle pipeline-level errors from vector search
    if isinstance(vector_context, str) and vector_context.startswith("[PIPELINE ERROR]"):
        error_detail = vector_context.replace("[PIPELINE ERROR] ", "")
        if "Access Denied" in error_detail or "unauthorized" in error_detail.lower():
            raise HTTPException(status_code=403, detail=error_detail)
        raise HTTPException(status_code=400, detail=error_detail)

    # 3. Check if no authorized data was found
    try:
        parsed = json.loads(vector_context)
        if not parsed:
            raise HTTPException(
                status_code=404,
                detail="No authorized data found for your query. Access may be restricted."
            )
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=500, detail="Unexpected response from vector search.")

    # 4. Detect role for response metadata
    user_role = "Unknown"
    try:
        user_role = parsed[0].get("metadata", {}).get("discipline", "Unknown")
    except Exception:
        pass

    # 5. Initialize Groq client
    groq_client = get_groq_client()
    if not groq_client:
        raise HTTPException(
            status_code=500,
            detail=(
                "GROQ_API_KEY is missing. Please set it in your .env file or terminal "
                "before sending requests."
            )
        )

    # 6. Build the secure system prompt from retrieved vector context
    system_prompt = (
        "You are an expert Secure SOP Agent for a General Contractor company. "
        "Your role is to answer questions STRICTLY based on the verified, role-restricted corporate data provided below. "
        "Always be professional, concise, and clear. "
        "Do NOT reveal data outside the provided context. "
        "If the user asks for something not present in the data, politely state that it is not available or not authorized."
    )

    user_message = (
        f"[VERIFIED SECURE DATA CONTEXT - Retrieved via RLS-filtered Vector Search]:\n"
        f"{vector_context}\n\n"
        f"---\n"
        f"USER QUERY: {req.message}"
    )

    # 7. Call Groq (Llama 3.3) for humanized answer
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=1024
        )
        llama_response = chat_completion.choices[0].message.content.strip()

    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch response from Groq API: {str(e)}"
        )

    # 8. Return production-grade structured response
    return {
        "status": "success",
        "user_id": user_id_val,
        "user_role": user_role,
        "reply": llama_response
    }


# Health Check
@app.get("/health")
def health_check():
    groq_status = "configured" if os.environ.get("GROQ_API_KEY") else "missing"
    return {
        "status": "healthy",
        "service": "Secure SOP Agent v3.0 (ChromaDB + Groq)",
        "groq_integration": groq_status
    }
