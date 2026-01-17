# =========================================================
# StartHub SOP Agent — Streamlit (FIXED & STABLE)
# =========================================================

import os, re, json, uuid
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

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

# =========================================================
# CONFIG
# =========================================================
INTENT_XLSX = "Agent_Mapping_11.xlsx"
GATE_KNOWLEDGE_XLSX = "gate_knowledge.xlsx"
GATE_REQUIREMENTS_XLSX = "gate_requirements.xlsx"

INTENT_DB = "./chroma_intents"
GATE_DB = "./chroma_gates"

MODEL = "gpt-4o-mini"
EMBEDDINGS = "text-embedding-3-large"

# =========================================================
# UTILS
# =========================================================
def normalize(t):
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
        gates.setdefault(gate, {"rules": [], "required_data": []})

        gates[gate]["rules"].append(Document(
            page_content=f"""
GATE: {gate}
SATISFIED WHEN:
{r['Satisfied When']}
""".strip(),
            metadata={"gate": gate}
        ))

        if "Required Data" in r and not pd.isna(r["Required Data"]):
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
gate_docs = [d for g in gate_requirements.values() for d in g["rules"]]
gate_vs = Chroma.from_documents(gate_docs, embeddings, persist_directory=GATE_DB)
gate_retriever = gate_vs.as_retriever(search_kwargs={"k": 1})

gate_knowledge = load_gate_knowledge()

# =========================================================
# LLM
# =========================================================
llm = ChatOpenAI(model=MODEL, temperature=0)

ASK_PROMPT = PromptTemplate.from_template("""
GATE RULES:
{rules}

Purpose:
{purpose}

Ask ONE question to satisfy the gate.
Return only the question.
""")

JUDGE_PROMPT = PromptTemplate.from_template("""
GATE RULES:
{rules}

User input:
{input}

Return JSON:
{{"satisfied": true/false, "reason": "..."}}
""")

def get_gate_rules(gate):
    docs = gate_retriever.invoke(gate)
    return docs[0].page_content if docs else ""

def ask_gate(gate):
    return (ASK_PROMPT | llm | StrOutputParser()).invoke({
        "rules": get_gate_rules(gate),
        "purpose": gate_knowledge[gate]["purpose"]
    })

def judge_gate(gate, text):
    raw = (JUDGE_PROMPT | llm | StrOutputParser()).invoke({
        "rules": get_gate_rules(gate),
        "input": text
    })
    return safe_json(raw, {"satisfied": False, "reason": "Unable to validate"})

# =========================================================
# STREAMLIT STATE
# =========================================================
st.set_page_config("StartHub SOP Agent", layout="wide")
st.title("🧠 StartHub SOP Agent")

if "state" not in st.session_state:
    st.session_state.state = {
        "conversation_id": str(uuid.uuid4()),
        "intent": None,
        "gate_order": [],
        "current_gate": None,
        "collected": {},
        "messages": [],
        "logs": []
    }

state = st.session_state.state

# =========================================================
# CHAT INPUT
# =========================================================
user_input = st.chat_input("Ask about your mail…")

if user_input:
    # 1️⃣ render user message
    st.chat_message("user").markdown(user_input)
    state["messages"].append(("user", user_input))

    # 2️⃣ resolve intent if needed
    if not state["intent"]:
        intent_doc = intent_retriever.invoke(normalize(user_input))[0]
        state["intent"] = intent_doc
        state["gate_order"] = [
            g.strip() for g in intent_doc.metadata["gate_order"].split(">")
        ]
        state["current_gate"] = state["gate_order"][0]

    gate = state["current_gate"]
    state["collected"].setdefault(gate, "")

    combined = (state["collected"][gate] + " " + user_input).strip()
    result = judge_gate(gate, combined)

    # LOG
    state["logs"].append({
        "timestamp": datetime.utcnow().isoformat(),
        "gate": gate,
        "input": user_input,
        "result": result
    })

    # 3️⃣ decision
    if result["satisfied"]:
        state["collected"][gate] = combined
        idx = state["gate_order"].index(gate)
        if idx + 1 < len(state["gate_order"]):
            state["current_gate"] = state["gate_order"][idx + 1]
            response = "Proceeding to the next step."
        else:
            response = "✅ All requirements met. Executing request."
    else:
        response = ask_gate(gate)

    # 4️⃣ render assistant message
    st.chat_message("assistant").markdown(response)
    state["messages"].append(("assistant", response))

# =========================================================
# LOG VIEWER
# =========================================================
with st.expander("📜 Logs"):
    st.json(state["logs"])
