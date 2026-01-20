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
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# STATE
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

# =========================================================
# LOAD INTENTS
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

# =========================================================
# LOAD GATE KNOWLEDGE
# =========================================================
def load_gate_knowledge():
    df = pd.read_excel(GATE_KNOWLEDGE_XLSX)
    return {
        r["Gate Name"]: {
            "purpose": r["Purpose"],
            "do_not_ask": r["Notes / Do Not Ask"]
        }
        for _, r in df.iterrows()
    }

# =========================================================
# LOAD GATE REQUIREMENTS (PATH-AWARE)
# =========================================================
def load_gate_requirements():
    df = pd.read_excel(GATE_REQUIREMENTS_XLSX)
    gates = {}

    for _, r in df.iterrows():
        gate = str(r["Gate Name"]).strip()
        gates.setdefault(gate, {
            "paths": [],
            "required_data": []
        })

        gates[gate]["paths"].append({
            "path": str(r["Satisfaction Paths"]).strip(),
            "satisfied_when": str(r["Satisfied When"]).strip(),
            "interpretation": str(r.get("Normalization / Interpretation Rules", "")).strip()
        })

        if not pd.isna(r.get("Required Data")):
            gates[gate]["required_data"].extend(
                [x.strip() for x in str(r["Required Data"]).split(";")]
            )

    return gates

# =========================================================
# VECTOR STORES
# =========================================================
embeddings = OpenAIEmbeddings(model=EMBEDDINGS, api_key=OPENAI_API_KEY)

intent_vs = Chroma.from_documents(load_intents(), embeddings, persist_directory=INTENT_DB)
intent_retriever = intent_vs.as_retriever(search_kwargs={"k": 1})

gate_requirements = load_gate_requirements()

gate_docs = []
for gate, g in gate_requirements.items():
    for p in g["paths"]:
        gate_docs.append(Document(
            page_content=p["path"],
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
PATH_SELECT_PROMPT = PromptTemplate.from_template("""
Select the best satisfaction path.

Gate: {gate}

Paths:
{paths}

User Input:
{input}

Return JSON only:
{{"path_index": number}}
""")

PATH_JUDGE_PROMPT = PromptTemplate.from_template("""
Satisfied When:
{rule}

User Provided:
{input}

Return JSON:
{{"satisfied": true/false, "reason": "..."}}
""")

ASK_PROMPT = PromptTemplate.from_template("""
Purpose:
{purpose}

Do NOT ask:
{do_not_ask}

Ask the best single question.
Return only the question.
""")

# =========================================================
# GATE ENGINE (PATH-AWARE)
# =========================================================
def select_path(gate, user_input):
    paths = gate_requirements[gate]["paths"]
    path_text = "\n".join(f"{i}. {p['path']}" for i, p in enumerate(paths))

    raw = (PATH_SELECT_PROMPT | llm | StrOutputParser()).invoke({
        "gate": gate,
        "paths": path_text,
        "input": user_input
    })

    idx = safe_json(raw, {"path_index": 0})["path_index"]
    return min(idx, len(paths) - 1)

def judge_path(path_rule, user_input):
    raw = (PATH_JUDGE_PROMPT | llm | StrOutputParser()).invoke({
        "rule": path_rule["satisfied_when"],
        "input": user_input
    })
    return safe_json(raw, {"satisfied": False, "reason": "Unable to validate"})

def ask_gate(gate):
    return (ASK_PROMPT | llm | StrOutputParser()).invoke({
        "purpose": gate_knowledge[gate]["purpose"],
        "do_not_ask": gate_knowledge[gate]["do_not_ask"]
    })

# =========================================================
# LOGGING
# =========================================================
def log_turn(log):
    print("\n==============================")
    print("CONVERSATIONAL AGENT LOG (v1)")
    print("==============================\n")
    for k, v in log.items():
        if isinstance(v, list):
            print(f"{k}:")
            for i in v:
                print(f"- {i}")
        else:
            print(f"{k}: {v}")
    print()

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
            "collected": {g: "" for g in gate_order}
        }

    session = SESSIONS[cid]
    gate = session["gate_order"][session["gate_idx"]]
    prev = session["collected"][gate]
    combined = (prev + " " + user_msg).strip()

    paths = gate_requirements[gate]["paths"]
    path_idx = select_path(gate, combined)

    satisfied = False
    reason = ""
    question = None

    while path_idx < len(paths):
        result = judge_path(paths[path_idx], combined)
        reason = result["reason"]

        if result["satisfied"]:
            satisfied = True
            break

        if "next path" in paths[path_idx]["interpretation"].lower():
            path_idx += 1
            continue

        question = ask_gate(gate)
        break

    log = {
        "Turn ID": datetime.utcnow().isoformat(),
        "Conversation ID": cid,
        "Customer Message": user_msg,
        "Intent": session["intent"].metadata["intent_id"],
        "Gate Order": session["gate_order"],
        "Current Gate": gate,
        "Data Read (expected)": gate_requirements[gate]["required_data"],
        "Collected Fields": {gate: combined if satisfied else prev},
        "Gate Evaluation Summary": reason,
        "Gate Status": "PASSED" if satisfied else "PENDING"
    }

    if satisfied:
        session["collected"][gate] = combined
        session["gate_idx"] += 1

        if session["gate_idx"] >= len(session["gate_order"]):
            log["Agent Decision"] = "EXECUTED_ACTION"
            log_turn(log)
            return ChatResponse(
                conversation_id=cid,
                agent_message="✅ All gates satisfied. Request executed.",
                done=True,
                log=log
            )

    if not question:
        question = ask_gate(gate)

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
