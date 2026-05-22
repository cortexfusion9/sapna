import os
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncpg
from groq import Groq
from contextlib import asynccontextmanager
from vector_service import semantic_secure_search, ingest_project_data

load_dotenv()

# --- LIFESPAN HANDLER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    db_pool = await asyncpg.create_pool(
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "Sapna123"),
        database="gc_subsc",
        host=os.environ.get("DB_HOST", "127.0.0.1"),
        port=5432
    )
    print("🚀 [STARTUP] Database connection ready.")
    ingest_project_data()
    yield
    await db_pool.close()
    print("🛑 [SHUTDOWN] Connection closed.")

app = FastAPI(title="General Contractor SaaS Agent", version="4.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str

class LoginPayload(BaseModel):
    username: str

# --- LOGIN ENDPOINT (FIXED 404) ---
@app.post("/api/auth/login")
async def secure_login(payload: LoginPayload):
    async with db_pool.acquire() as conn:
        user = await conn.fetchrow("SELECT username FROM users WHERE username = $1", payload.username.strip())
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return {"status": "authenticated", "token": user["username"]}

# --- CHAT ENDPOINT ---
async def check_subscription_access(username: str, module: str) -> bool:
    async with db_pool.acquire() as conn:
        query = "SELECT count(*) FROM user_subscriptions s JOIN users u ON s.user_id = u.user_id WHERE u.username = $1 AND s.module_name = $2 AND s.is_active = TRUE"
        return await conn.fetchval(query, username, module) > 0

@app.post("/api/chat")
async def secure_chat_endpoint(req: ChatRequest, x_acting_as: str = Header(..., alias="X-Acting-As")):
    if not await check_subscription_access(x_acting_as, 'MORGAN'):
        raise HTTPException(status_code=403, detail="Access Denied.")

    sql_context = ""
    try:
        async with db_pool.acquire() as conn:
            p = await conn.fetch("SELECT project_name, budget, progress FROM pm_projects p JOIN users u ON p.user_id = u.user_id WHERE u.username = $1", x_acting_as)
            d = await conn.fetch("SELECT client_name, deal_value, status FROM crm_deals d JOIN users u ON d.user_id = u.user_id WHERE u.username = $1", x_acting_as)
            b = await conn.fetch("SELECT proposal_name, amount, status FROM bid_management b JOIN users u ON b.user_id = u.user_id WHERE u.username = $1", x_acting_as)
            sql_context = f"Projects: {str([dict(r) for r in p])}\nDeals: {str([dict(r) for r in d])}\nBids: {str([dict(r) for r in b])}"
    except:
        sql_context = "No data found."

    vector_context = await semantic_secure_search(user_id=x_acting_as, user_query=req.message)
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    system_instruction = """
    You are Morgan, a professional construction assistant.
    - YOUR GOAL: Be helpful but precise.
    - CLARIFICATION RULE: If a user asks a general or ambiguous question (e.g., "What is the highest bid?"), 
      DO NOT guess. Instead, ask the user to clarify: 
      "Are you asking about the highest bid among your own personal proposals, or are you looking for the highest bid in the entire database?"
    - ONLY provide data once the user specifies their intent.
    - If the user provides enough context (e.g., "What is the highest bid in my account?"), provide the exact data from the [Context].
    - Maintain a warm, proactive, and professional tone.
    """
    
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Context: {sql_context}\n\nInfo: {vector_context}\n\nQuery: {req.message}"}
        ],
        model="llama-3.3-70b-versatile"
    )
    
    return {"status": "success", "reply": chat_completion.choices[0].message.content}

@app.get("/")
def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())