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
        gate = r["Gate Name"]

        gates.setdefault(gate, {
            "paths": [],
            "required_data": []
        })

        # ---- Parse Satisfaction Paths ----
        raw_paths = str(r["Satisfaction Paths"]).strip()
        for line in raw_paths.split("\n"):
            if ":" not in line:
                continue

            label, desc = line.split(":", 1)

            gates[gate]["paths"].append({
                "path": label.strip(),
                "description": desc.strip(),
                "satisfied_when": str(r["Satisfied When"]).strip(),
                "interpretation": str(
                    r.get("Normalization / Interpretation Rules", "")
                ).strip()
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
        gate_docs.append(
            Document(
                page_content=f"""
PATH: {p['path']}
DESCRIPTION: {p['description']}
""".strip(),
                metadata={"gate": gate}
            )
        )

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
You are a professional customer support agent.

ACTIVE PATH:
{path_description}

Purpose:
{purpose}

Do NOT ask:
{do_not_ask}

Ask ONE best question to satisfy THIS path.
Return only the question.
""")

JUDGE_PROMPT = PromptTemplate.from_template("""
PATH DESCRIPTION:
{path_description}

SATISFIED WHEN:
{satisfied_when}

INTERPRETATION RULES:
{interpretation}

USER INPUT:
{input}

Return JSON:
{{"satisfied": true/false, "reason": "..."}}
""")

PATH_SELECT_PROMPT = PromptTemplate.from_template("""
User message:
{input}

Available paths:
{paths}

Select the most relevant path.
Return JSON only:
{{"path": "Path A"}}
""")

# =========================================================
# HELPERS
# =========================================================
def select_path(gate, user_input):
    paths = gate_requirements[gate]["paths"]
    raw = (PATH_SELECT_PROMPT | llm | StrOutputParser()).invoke({
        "input": user_input,
        "paths": json.dumps(paths, indent=2)
    })
    parsed = safe_json(raw, {})
    for i, p in enumerate(paths):
        if p["path"] == parsed.get("path"):
            return i
    return 0

def judge_path(path, text):
    raw = (JUDGE_PROMPT | llm | StrOutputParser()).invoke({
        "path_description": path["description"],
        "satisfied_when": path["satisfied_when"],
        "interpretation": path["interpretation"],
        "input": text
    })
    return safe_json(raw, {"satisfied": False, "reason": "Unable to validate"})

def ask_for_path(gate, path):
    return (ASK_PROMPT | llm | StrOutputParser()).invoke({
        "path_description": path["description"],
        "purpose": gate_knowledge[gate]["purpose"],
        "do_not_ask": gate_knowledge[gate]["do_not_ask"]
    })

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
        intent = intent_retriever.invoke(normalize(user_msg))[0]
        gate_order = [g.strip() for g in intent.metadata["gate_order"].split(">")]

        SESSIONS[cid] = {
            "intent": intent,
            "gate_order": gate_order,
            "gate_idx": 0,
            "path_idx": None,
            "collected": {g: "" for g in gate_order}
        }

    session = SESSIONS[cid]
    gate = session["gate_order"][session["gate_idx"]]

    if session["path_idx"] is None:
        session["path_idx"] = select_path(gate, user_msg)

    path = gate_requirements[gate]["paths"][session["path_idx"]]
    combined = (session["collected"][gate] + " " + user_msg).strip()

    result = judge_path(path, combined)

    log = {
        "Turn ID": datetime.utcnow().isoformat(),
        "Conversation ID": cid,
        "Customer Message": user_msg,
        "Intent": session["intent"].metadata["intent_id"],
        "Current Gate": gate,
        "Current Path": path["path"],
        "Path Description": path["description"],
        "Gate Status": "PASSED" if result["satisfied"] else "FAILED",
        "Evaluation Reason": result["reason"]
    }

    if result["satisfied"]:
        session["collected"][gate] = combined
        session["gate_idx"] += 1
        session["path_idx"] = None

        if session["gate_idx"] >= len(session["gate_order"]):
            log["Agent Decision"] = "EXECUTED_ACTION"
            log_turn(log)
            return ChatResponse(
                conversation_id=cid,
                agent_message="✅ All gates satisfied. Request executed.",
                done=True,
                log=log
            )

    else:
        session["path_idx"] += 1
        if session["path_idx"] >= len(gate_requirements[gate]["paths"]):
            log["Agent Decision"] = "ESCALATION_REQUIRED"
            log_turn(log)
            return ChatResponse(
                conversation_id=cid,
                agent_message="I’m unable to identify the correct item. A support agent will review this.",
                done=True,
                log=log
            )

    question = ask_for_path(gate, gate_requirements[gate]["paths"][session["path_idx"]])
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
