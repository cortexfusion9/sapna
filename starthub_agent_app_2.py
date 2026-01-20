import os, re, json, uuid
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =========================================================
# ENV
# =========================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")

# =========================================================
# CONFIG
# =========================================================
INTENT_XLSX = "Agent_Mapping_11.xlsx"
GATE_KNOWLEDGE_XLSX = "gate_knowledge.xlsx"
GATE_REQUIREMENTS_XLSX = "gate_requirements.xlsx"

INTENT_DB = "./chroma_intents"
GATE_DB = "./chroma_gates"

EMBEDDINGS = "text-embedding-3-large"
MODEL = "gpt-4o-mini"  # ⚠️ use real model

# =========================================================
# APP
# =========================================================
app = FastAPI(title="StartHub SOP Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS = {}

# =========================================================
# UTILS
# =========================================================
def normalize(t: str):
    return re.sub(r"[^\w\s]", "", t.lower()).strip()

def safe_json(t, default):
    try:
        return json.loads(t)
    except:
        return default

# =========================================================
# LOAD DATA
# =========================================================
def load_intents():
    df = pd.read_excel(INTENT_XLSX)
    docs = []
    for _, r in df.iterrows():
        docs.append(Document(
            page_content=f"""
INTENT:
{r['Intent ID (input)']}
REQUEST_VARIATIONS:
{r['Request Variations']}
""".strip(),
            metadata={
                "intent_id": r["Intent ID (input)"],
                "gate_order": r["GATE ORDER"],
            }
        ))
    return docs

def load_gate_knowledge():
    df = pd.read_excel(GATE_KNOWLEDGE_XLSX)
    return {
        r["Gate Name"]: {
            "purpose": r["Purpose"],
            "do_not_ask": r["Notes / Do Not Ask"]
        } for _, r in df.iterrows()
    }

def load_gate_requirements():
    df = pd.read_excel(GATE_REQUIREMENTS_XLSX)
    gates = {}

    for _, r in df.iterrows():
        gate = r["Gate Name"]
        gates.setdefault(gate, [])

        for line in str(r["Satisfaction Paths"]).split("\n"):
            if ":" not in line:
                continue
            label, desc = line.split(":", 1)
            gates[gate].append({
                "path": label.strip(),
                "description": desc.strip(),
                "satisfied_when": r["Satisfied When"],
                "interpretation": r.get("Normalization / Interpretation Rules", "")
            })
    return gates

# =========================================================
# VECTOR STORES
# =========================================================
embeddings = OpenAIEmbeddings(model=EMBEDDINGS)

intent_vs = Chroma.from_documents(load_intents(), embeddings, persist_directory=INTENT_DB)
intent_retriever = intent_vs.as_retriever(k=1)

gate_requirements = load_gate_requirements()
gate_knowledge = load_gate_knowledge()

# =========================================================
# LLM
# =========================================================
llm = ChatOpenAI(model=MODEL, temperature=0)

# =========================================================
# PROMPTS
# =========================================================
ASK_PROMPT = PromptTemplate.from_template("""
ACTIVE PATH:
{path}

Purpose:
{purpose}

Ask ONE question to satisfy this path.
""")

JUDGE_PROMPT = PromptTemplate.from_template("""
PATH:
{path}

SATISFIED WHEN:
{satisfied_when}

INTERPRETATION:
{interpretation}

INPUT:
{input}

Return JSON:
{{"satisfied": true/false, "reason": "..."}}
""")

# =========================================================
# HELPERS
# =========================================================
def ask_for_path(gate, path):
    return (ASK_PROMPT | llm | StrOutputParser()).invoke({
        "path": path["description"],
        "purpose": gate_knowledge[gate]["purpose"]
    })

def judge_path(path, text):
    raw = (JUDGE_PROMPT | llm | StrOutputParser()).invoke({
        "path": path["description"],
        "satisfied_when": path["satisfied_when"],
        "interpretation": path["interpretation"],
        "input": text
    })
    return safe_json(raw, {"satisfied": False})

# =========================================================
# LOGGING
# =========================================================
def log_turn(log):
    print("\n==============================")
    print("CONVERSATIONAL AGENT LOG (v1)")
    print("==============================")
    for k, v in log.items():
        print(f"{k}: {v}")
    print()

# =========================================================
# API
# =========================================================
class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None

class ChatResponse(BaseModel):
    conversation_id: str
    agent_message: str
    done: bool
    log: dict

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    cid = req.conversation_id or str(uuid.uuid4())

    if cid not in SESSIONS:
        intent = intent_retriever.invoke(normalize(req.message))[0]
        gate_order = intent.metadata["gate_order"].split(">")
        SESSIONS[cid] = {
            "intent": intent.metadata["intent_id"],
            "gate_order": gate_order,
            "gate_idx": 0,
            "path_idx": 0,
            "path_asked": False,
            "collected": {g: "" for g in gate_order}
        }

    s = SESSIONS[cid]
    gate = s["gate_order"][s["gate_idx"]]
    paths = gate_requirements[gate]
    path = paths[s["path_idx"]]

    log = {
        "Turn ID": datetime.utcnow().isoformat(),
        "Gate": gate,
        "Path": path["path"],
        "Customer Message": req.message
    }

    if not s["path_asked"]:
        q = ask_for_path(gate, path)
        s["path_asked"] = True
        log["Decision"] = "ASK"
        log_turn(log)
        return ChatResponse(conversation_id = cid, 
                            agent_message=q, 
                            done=False, 
                            log=log)

    combined = s["collected"][gate] + " " + req.message
    result = judge_path(path, combined)

    if result["satisfied"]:
        s["collected"][gate] = combined
        s["gate_idx"] += 1
        s["path_idx"] = 0
        s["path_asked"] = False
        log["Decision"] = "PATH_SATISFIED"
    else:
        s["path_idx"] += 1
        s["path_asked"] = False
        log["Decision"] = "NEXT_PATH"

        if s["path_idx"] >= len(paths):
            log["Decision"] = "ESCALATE"
            log_turn(log)
            return ChatResponse(conversation_id =cid, 
                                agent_message ="I’m unable to identify the correct item. A support agent will review this.", 
                                done=True, 
                                log=log)

    question = ask_for_path(gate, paths[s["path_idx"]])
    log["Agent Decision"] = "Requested more information"
    log["Agent Message"] = question
    
    log_turn(log)
    return ChatResponse(conversation_id = cid, 
                        agent_message = question,
                        done = False,
                        log = log)

@app.get("/health")
def health():
    return {"status": "ok"}


