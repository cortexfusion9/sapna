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

SESSIONS = {}

# =========================================================
# UTILS
# =========================================================
def normalize(text: str):
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

def is_unknown(text: str):
    return normalize(text) in {
        "i dont know", "dont know", "not sure", "no idea", "unknown"
    }

def safe_json(txt, default):
    try:
        return json.loads(txt)
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
            "paths": [],
            "required_data": []
        })

        gates[gate]["paths"].append({
            "path": r["Satisfaction Paths"],
            "satisfied_when": r["Satisfied When"],
            "interpretation": r.get("Interpretation / Normalization Rules", ""),
            "question": r.get("Follow-up Question Templates", "")
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
            page_content=f"""
PATH: {p['path']}
SATISFIED WHEN:
{p['satisfied_when']}
INTERPRETATION:
{p['interpretation']}
""",
            metadata={"gate": gate, "path": p["path"]}
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
JUDGE_PROMPT = PromptTemplate.from_template("""
SATISFIED WHEN:
{satisfied_when}

INTERPRETATION RULES:
{interpretation}

User Input:
{input}

Return JSON:
{{"satisfied": true/false, "reason": "..."}}
""")

# =========================================================
# CORE ORCHESTRATION
# =========================================================
def judge_path(rule, user_input):
    if is_unknown(user_input):
        return {
            "satisfied": False,
            "reason": "User explicitly does not know → path exhausted"
        }

    raw = (JUDGE_PROMPT | llm | StrOutputParser()).invoke({
        "satisfied_when": rule["satisfied_when"],
        "interpretation": rule["interpretation"],
        "input": user_input
    })
    return safe_json(raw, {"satisfied": False, "reason": "Unable to validate"})

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
    paths = gate_requirements[gate]["paths"]
    pidx = s["path_idx"][gate]

    # Exhausted all paths
    if pidx >= len(paths):
        return ChatResponse(
            conversation_id=cid,
            agent_message="I can’t identify the required information. I will escalate this request.",
            done=False,
            log={"Gate": gate, "Status": "ESCALATED"}
        )

    rule = paths[pidx]
    combined = (s["collected"][gate] + " " + user_msg).strip()

    result = judge_path(rule, combined)

    log = {
        "Turn ID": datetime.utcnow().isoformat(),
        "Conversation ID": cid,
        "Intent": s["intent"].metadata["intent_id"],
        "Gate": gate,
        "Path": rule["path"],
        "Path Index": pidx,
        "Collected": combined,
        "Path Result": result["reason"]
    }

    # Path satisfied
    if result["satisfied"]:
        s["collected"][gate] = combined
        s["gate_idx"] += 1
        s["path_idx"][gate] = 0

        if s["gate_idx"] >= len(s["gate_order"]):
            return ChatResponse(
                conversation_id=cid,
                agent_message="✅ All gates satisfied. Request executed.",
                done=True,
                log=log
            )

        return ChatResponse(
            conversation_id=cid,
            agent_message="Proceeding to the next step.",
            done=False,
            log=log
        )

    # Path failed → move to next path
    s["path_idx"][gate] += 1

    # Ask next path question
    if s["path_idx"][gate] < len(paths):
        q = paths[s["path_idx"][gate]]["question"]
        return ChatResponse(
            conversation_id=cid,
            agent_message=q,
            done=False,
            log=log
        )

    return ChatResponse(
        conversation_id=cid,
        agent_message="Unable to proceed with the information provided. Escalating.",
        done=False,
        log=log
    )

@app.get("/health")
def health():
    return {"status": "ok"}
