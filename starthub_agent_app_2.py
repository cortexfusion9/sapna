import os, re, json, uuid
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from difflib import get_close_matches

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware

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

SESSIONS = {}

# =========================================================
# UTILS
# =========================================================
def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

def fuzzy_normalize(value: str, vocabulary: list[str]) -> str:
    """Fix misspellings using fuzzy match"""
    matches = get_close_matches(value.lower(), vocabulary, n=1, cutoff=0.8)
    return matches[0] if matches else value

def safe_json(txt, default):
    try:
        return json.loads(txt)
    except:
        return default

def is_unknown_response(text: str) -> bool:
    return normalize(text) in {"dont know", "i dont know", "no idea", "unknown"}

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
""",
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
            "path": r["Satisfaction Paths"],
            "satisfied_when": r["Satisfied When"],
            "interpretation": r.get("Normalization / Interpretation Rules", ""),
            "question": r.get("Follow-up Question Templates", "")
        })
    return gates

# =========================================================
# VECTOR STORES
# =========================================================
embeddings = OpenAIEmbeddings(model=EMBEDDINGS, api_key=OPENAI_API_KEY)

intent_vs = Chroma.from_documents(load_intents(), embeddings, persist_directory=INTENT_DB)
intent_retriever = intent_vs.as_retriever(search_kwargs={"k": 1})

gate_requirements = load_gate_requirements()
gate_docs = []
for gate, rules in gate_requirements.items():
    for r in rules:
        gate_docs.append(Document(
            page_content=f"""
PATH:
{r['path']}

SATISFIED WHEN:
{r['satisfied_when']}

INTERPRETATION RULES:
{r['interpretation']}
""",
            metadata={"gate": gate}
        ))

gate_vs = Chroma.from_documents(gate_docs, embeddings, persist_directory=GATE_DB)
gate_retriever = gate_vs.as_retriever(search_kwargs={"k": 1})

gate_knowledge = load_gate_knowledge()

# =========================================================
# LLM
# =========================================================
llm = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY, temperature=0)

# =========================================================
# PROMPTS
# =========================================================
ASK_PROMPT = PromptTemplate.from_template("""
You are a customer support agent.

CURRENT PATH:
{path}

Purpose:
{purpose}

Ask the BEST question for this path.
Return only the question.
""")

JUDGE_PROMPT = PromptTemplate.from_template("""
SATISFIED WHEN:
{rule}

User input:
{input}

Return JSON:
{{"satisfied": true/false, "reason": "..."}}
""")

# =========================================================
# PATH ENGINE
# =========================================================
def judge_path(rule, user_input):
    if is_unknown_response(user_input):
        return {"satisfied": False, "reason": "User explicitly does not know → move to next path"}

    raw = (JUDGE_PROMPT | llm | StrOutputParser()).invoke({
        "rule": rule["satisfied_when"],
        "input": user_input
    })
    return safe_json(raw, {"satisfied": False, "reason": "Unable to validate"})

def ask_path(gate, rule):
    return (ASK_PROMPT | llm | StrOutputParser()).invoke({
        "path": rule["path"],
        "purpose": gate_knowledge[gate]["purpose"]
    })

# =========================================================
# API SCHEMAS
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
    user_msg = req.message

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
    path_idx = s["path_idx"][gate]
    paths = gate_requirements[gate]

    # If paths exhausted → fail safe
    if path_idx >= len(paths):
        s["gate_idx"] += 1
        return ChatResponse(
            conversation_id=cid,
            agent_message="I’ll proceed with the request using available information.",
            done=False,
            log={"Gate": gate, "Path": "EXHAUSTED"}
        )

    rule = paths[path_idx]
    combined = (s["collected"][gate] + " " + user_msg).strip()
    result = judge_path(rule, combined)

    log = {
        "Turn ID": datetime.utcnow().isoformat(),
        "Conversation ID": cid,
        "Intent": s["intent"].metadata["intent_id"],
        "Gate": gate,
        "Path Attempted": rule["path"],
        "Path Index": path_idx,
        "Collected": combined,
        "Path Result": "PASSED" if result["satisfied"] else "FAILED",
        "Reason": result["reason"]
    }

    if result["satisfied"]:
        s["collected"][gate] = combined
        s["gate_idx"] += 1
        log["Decision"] = "GATE_PASSED"
        return ChatResponse(
            conversation_id=cid,
            agent_message="✅ Information confirmed.",
            done=s["gate_idx"] >= len(s["gate_order"]),
            log=log
        )

    # Path failed → move to NEXT PATH
    s["path_idx"][gate] += 1
    next_question = ask_path(gate, paths[s["path_idx"][gate]]) \
        if s["path_idx"][gate] < len(paths) else \
        "I will proceed with the request."

    log["Decision"] = "MOVE_TO_NEXT_PATH"

    return ChatResponse(
        conversation_id=cid,
        agent_message=next_question,
        done=False,
        log=log
    )

@app.get("/health")
def health():
    return {"status": "ok"}
