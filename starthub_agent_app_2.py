import os, re, json, uuid
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

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
    allow_origins=["*"],  # or restrict to your static site URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =========================================================
# STATE (Render-safe single instance)
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
        gates.setdefault(gate, {
            "rules": [],
            "required_data": []
        })

        gates[gate]["rules"].append(f"""
SATISFACTION PATHS:
{r['Satisfaction Paths']}

SATISFIED WHEN:
{r['Satisfied When']}
""")

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
    for rule in g["rules"]:
        gate_docs.append(Document(page_content=rule, metadata={"gate": gate}))

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
GATE RULES:
{rules}

Purpose:
{purpose}

Do NOT ask:
{do_not_ask}

Ask ONE best question.
Return only the question.
""")

JUDGE_PROMPT = PromptTemplate.from_template("""
GATE RULES:
{rules}

User provided:
{input}

Return JSON:
{{"satisfied": true/false, "reason": "..."}}
""")

# =========================================================
# HELPERS
# =========================================================
def get_gate_rules(gate):
    docs = gate_retriever.invoke(gate)
    return docs[0].page_content if docs else ""

def ask_gate(gate):
    return (ASK_PROMPT | llm | StrOutputParser()).invoke({
        "rules": get_gate_rules(gate),
        "purpose": gate_knowledge[gate]["purpose"],
        "do_not_ask": gate_knowledge[gate]["do_not_ask"]
    })

def judge_gate(gate, text):
    raw = (JUDGE_PROMPT | llm | StrOutputParser()).invoke({
        "rules": get_gate_rules(gate),
        "input": text
    })
    return safe_json(raw, {"satisfied": False, "reason": "Unable to validate"})

# =========================================================
# LOGGING
# =========================================================
def log_turn(**log):
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

    result = judge_gate(gate, combined)
    required_data = gate_requirements[gate]["required_data"]

    next_gate = (
        session["gate_order"][session["gate_idx"] + 1]
        if result["satisfied"] and session["gate_idx"] < len(session["gate_order"]) - 1
        else None
    )

    log = {
        "Turn ID": datetime.utcnow().isoformat(),
        "Conversation ID": cid,
        "Customer Message": user_msg,
        "Intent": session["intent"].metadata["intent_id"],
        "Gate Order": session["gate_order"],
        "Current Gate": gate,
        "Data Read (expected)": required_data,
        "Collected Fields": {gate: combined if result["satisfied"] else prev},
        "Gate Evaluation Summary": result["reason"],
        "Gate Status": "PASSED" if result["satisfied"] else "PENDING",
        "Next Gate": next_gate,
    }

    if result["satisfied"]:
        session["collected"][gate] = combined
        session["gate_idx"] += 1

        if session["gate_idx"] >= len(session["gate_order"]):
            log["Agent Decision"] = "EXECUTED_ACTION"
            log_turn(**log)
            return ChatResponse(
                conversation_id=cid,
                agent_message="✅ All gates satisfied. Request executed.",
                done=True,
                log=log
            )

    question = ask_gate(gate)
    log["Agent Decision"] = "REQUESTED_MORE_INFORMATION"
    log["Agent Message"] = question

    log_turn(**log)

    return ChatResponse(
        conversation_id=cid,
        agent_message=question,
        done=False,
        log=log
    )

@app.get("/health")
def health():
    return {"status": "ok"}


