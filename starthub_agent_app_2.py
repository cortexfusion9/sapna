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
MODEL = "gpt-5.2"

# =========================================================
# APP
# =========================================================
app = FastAPI(title="StartHub SOP Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# SESSION STATE
# =========================================================
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

def is_unknown(text: str):
    t = text.lower()
    return any(x in t for x in [
        "dont know", "do not know", "not sure", "no idea", "you have it"
    ])

# =========================================================
# LOAD DATA
# =========================================================
def load_intents():
    df = pd.read_excel(INTENT_XLSX)
    docs = []
    for _, r in df.iterrows():
        if pd.isna(r["SUPPORT AGENT ACTION FLOW"]):
            continue
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
                "action_flow": r["SUPPORT AGENT ACTION FLOW"]
            }
        ))
    return docs

def load_gate_knowledge():
    df = pd.read_excel(GATE_KNOWLEDGE_XLSX)
    return {
        r["Gate Name"]: {
            "purpose": r["Purpose"],
            "do_not_ask": r["Notes / Do Not Ask"]
        }
        for _, r in df.iterrows()
    }

def load_gate_requirements():
    df = pd.read_excel(GATE_REQUIREMENTS_XLSX)
    gates = {}

    for _, r in df.iterrows():
        gate = r["Gate Name"]
        gates.setdefault(gate, [])

        gates[gate].append({
            "path": r["Path"],
            "satisfaction_paths": r["Satisfaction Paths"],
            "satisfied_when": r["Satisfied When"],
            "interpretation": r.get("Normalization / Interpretation Rules", ""),
            "required_data": (
                [] if pd.isna(r.get("Required Data"))
                else [x.strip() for x in str(r["Required Data"]).split(";")]
            )
        })
    return gates

# =========================================================
# VECTOR STORES
# =========================================================
embeddings = OpenAIEmbeddings(model=EMBEDDINGS, api_key=OPENAI_API_KEY)

intent_vs = Chroma.from_documents(load_intents(), embeddings, persist_directory=INTENT_DB)
intent_retriever = intent_vs.as_retriever(search_kwargs={"k": 1})

gate_paths = load_gate_requirements()

gate_docs = []
for gate, paths in gate_paths.items():
    for p in paths:
        gate_docs.append(Document(
            page_content=f"""
PATH: {p['path']}
SATISFACTION PATHS:
{p['satisfaction_paths']}
SATISFIED WHEN:
{p['satisfied_when']}
INTERPRETATION:
{p['interpretation']}
""",
            metadata={"gate": gate, "path": p["path"]}
        ))

gate_vs = Chroma.from_documents(gate_docs, embeddings, persist_directory=GATE_DB)
gate_retriever = gate_vs.as_retriever(search_kwargs={"k": 3})

gate_knowledge = load_gate_knowledge()

# =========================================================
# LLM
# =========================================================
llm = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY, temperature=0)

# =========================================================
# PROMPTS
# =========================================================
ASK_PROMPT = PromptTemplate.from_template("""
You are a professional customer support agent following SOP.

GATE: {gate}
PATH: {path}

PATH GOAL:
{satisfied_when}

INTERPRETATION RULES:
{interpretation}

Gate Purpose:
{purpose}

Do NOT ask:
{do_not_ask}

Conversation so far:
{conversation}

Ask the single best next question to satisfy THIS PATH.
Return only the question.
""")

JUDGE_PROMPT = PromptTemplate.from_template("""
Evaluate ONLY this path.

PATH GOAL:
{satisfied_when}

INTERPRETATION RULES:
{interpretation}

User input:
{input}

Return JSON only:
{{"satisfied": true/false, "reason": "..."}}
""")

# =========================================================
# LOGGING
# =========================================================
def log_turn(log):
    print("\n==============================")
    print("CONVERSATIONAL AGENT LOG (v1)")
    print("==============================")
    for k, v in log.items():
        if isinstance(v, list):
            print(f"{k}:")
            for i in v:
                print(f"- {i}")
        else:
            print(f"{k}: {v}")
    print()

# =========================================================
# API MODELS
# =========================================================
class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None

class ChatResponse(BaseModel):
    conversation_id: str
    agent_message: str
    done: bool
    log: dict

# =========================================================
# CHAT ENDPOINT
# =========================================================
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    cid = req.conversation_id or str(uuid.uuid4())
    user_msg = req.message.strip()

    if cid not in SESSIONS:
        intent_doc = intent_retriever.invoke(normalize(user_msg))[0]
        gate_order = [g.strip() for g in intent_doc.metadata["gate_order"].split(">")]

        SESSIONS[cid] = {
            "intent": intent_doc,
            "gate_order": gate_order,
            "gate_idx": 0,
            "path_idx": {g: 0 for g in gate_order},
            "collected": {g: "" for g in gate_order}
        }

    s = SESSIONS[cid]
    gate = s["gate_order"][s["gate_idx"]]
    paths = gate_paths[gate]
    pidx = s["path_idx"][gate]

    prev = s["collected"][gate]
    combined = (prev + " " + user_msg).strip()

    path = paths[pidx]

    # ---------- JUDGE ----------
    judge = (JUDGE_PROMPT | llm | StrOutputParser()).invoke({
        "satisfied_when": path["satisfied_when"],
        "interpretation": path["interpretation"],
        "input": combined
    })
    result = safe_json(judge, {"satisfied": False, "reason": "Unable to validate"})

    # ---------- LOG ----------
    log = {
        "Turn ID": datetime.utcnow().isoformat(),
        "Conversation ID": cid,
        "Intent": s["intent"].metadata["intent_id"],
        "Gate Order": s["gate_order"],
        "Current Gate": gate,
        "Current Path": path["path"],
        "Data Read (expected)": path["required_data"],
        "Collected Fields": {gate: combined},
        "Gate Evaluation Summary": result["reason"],
        "Gate Status": "PASSED" if result["satisfied"] else "PENDING"
    }

    # ---------- PATH SATISFIED ----------
    if result["satisfied"]:
        s["collected"][gate] = combined
        s["gate_idx"] += 1

        if s["gate_idx"] >= len(s["gate_order"]):
            log["Agent Decision"] = "EXECUTED_ACTION"
            log_turn(log)
            return ChatResponse(
                conversation_id=cid,
                agent_message="✅ All gates satisfied. Request executed.",
                done=True,
                log=log
            )

    # ---------- PATH FAILED → MOVE ----------
    elif is_unknown(user_msg):
        s["path_idx"][gate] += 1
        if s["path_idx"][gate] >= len(paths):
            log["Agent Decision"] = "ESCALATION_REQUIRED"
            log_turn(log)
            return ChatResponse(
                conversation_id=cid,
                agent_message="I’m unable to identify the correct mail item. I’ll escalate this for manual assistance.",
                done=True,
                log=log
            )

    # ---------- ASK NEXT QUESTION ----------
    next_path = paths[s["path_idx"][gate]]
    question = (ASK_PROMPT | llm | StrOutputParser()).invoke({
        "gate": gate,
        "path": next_path["path"],
        "satisfied_when": next_path["satisfied_when"],
        "interpretation": next_path["interpretation"],
        "purpose": gate_knowledge[gate]["purpose"],
        "do_not_ask": gate_knowledge[gate]["do_not_ask"],
        "conversation": combined
    })

    log["Agent Decision"] = "REQUESTED_MORE_INFORMATION"
    log["Agent Message"] = question
    log_turn(log)

    return ChatResponse(
        conversation_id=cid,
        agent_message=question,
        done=False,
        log=log
    )

@app.get("/health")
def health():
    return {"status": "ok"}
