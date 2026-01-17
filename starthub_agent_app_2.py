# =========================================================
# StartHub SOP Agent — Streamlit Deployment (FINAL)
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
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing")
    st.stop()


# =========================================================
# CONFIG
# =========================================================
INTENT_XLSX = "Agent_Mapping_11.xlsx"
GATE_KNOWLEDGE_XLSX = "gate_knowledge.xlsx"
GATE_REQUIREMENTS_XLSX = "gate_requirements.xlsx"

INTENT_DB = "./chroma_intents"
GATE_DB = "./chroma_gates"

EMBEDDINGS = "text-embedding-3-large"
MODEL = "gpt-4o-mini"   # use stable model for prod


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
                "intent_id": str(r["Intent ID (input)"]),
                "gate_order": str(r["GATE ORDER"]),
                "action_flow": str(r["SUPPORT AGENT ACTION FLOW"])
            }
        ))
    return docs


# =========================================================
# LOAD GATE KNOWLEDGE
# =========================================================
def load_gate_knowledge():
    df = pd.read_excel(GATE_KNOWLEDGE_XLSX)
    return {
        str(r["Gate Name"]).strip(): {
            "purpose": str(r["Purpose"]),
            "do_not_ask": str(r["Notes / Do Not Ask"])
        }
        for _, r in df.iterrows()
    }


# =========================================================
# LOAD GATE REQUIREMENTS (WITH REQUIRED DATA)
# =========================================================
def load_gate_requirements():
    df = pd.read_excel(GATE_REQUIREMENTS_XLSX)
    gates = {}

    for _, r in df.iterrows():
        gate = str(r["Gate Name"]).strip()

        gates.setdefault(gate, {
            "rules": [],
            "required_data": set()
        })

        gates[gate]["rules"].append(Document(
            page_content=f"""
GATE: {gate}

SATISFACTION PATHS:
{r['Satisfaction Paths']}

SATISFIED WHEN:
{r['Satisfied When']}
""".strip(),
            metadata={"gate": gate}
        ))

        if "Required Data" in r and not pd.isna(r["Required Data"]):
            for f in str(r["Required Data"]).split(";"):
                gates[gate]["required_data"].add(f.strip())

    return gates


# =========================================================
# VECTOR STORES
# =========================================================
embeddings = OpenAIEmbeddings(model=EMBEDDINGS, api_key=OPENAI_API_KEY)

intent_vs = Chroma.from_documents(load_intents(), embeddings, persist_directory=INTENT_DB)
intent_retriever = intent_vs.as_retriever(search_kwargs={"k": 1})

gate_requirements = load_gate_requirements()

gate_docs = []
for g in gate_requirements.values():
    gate_docs.extend(g["rules"])

gate_vs = Chroma.from_documents(gate_docs, embeddings, persist_directory=GATE_DB)
gate_retriever = gate_vs.as_retriever(search_kwargs={"k": 1})

gate_knowledge = load_gate_knowledge()


# =========================================================
# HELPERS
# =========================================================
def get_gate_rules(gate):
    docs = gate_retriever.invoke(gate)
    return docs[0].page_content if docs else "No rules found."

def get_required_data(gate):
    return sorted(gate_requirements.get(gate, {}).get("required_data", []))


# =========================================================
# LLM
# =========================================================
llm = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY, temperature=0)


# =========================================================
# PROMPTS
# =========================================================
ASK_PROMPT = PromptTemplate.from_template("""
You are a professional customer support agent.

GATE RULES:
{gate_rules}

Purpose:
{purpose}

Do NOT ask:
{do_not_ask}

Ask the single best question to satisfy the gate.
Return only the question.
""")

JUDGE_PROMPT = PromptTemplate.from_template("""
You are a gate validation engine.

GATE RULES:
{gate_rules}

User has provided:
{collected}

Decide if the gate is satisfied.

Return JSON only:
{{"satisfied": true/false, "reason": "..."}}
""")


# =========================================================
# GATE OPERATIONS
# =========================================================
def ask_gate(gate):
    return (ASK_PROMPT | llm | StrOutputParser()).invoke({
        "gate_rules": get_gate_rules(gate),
        "purpose": gate_knowledge[gate]["purpose"],
        "do_not_ask": gate_knowledge[gate]["do_not_ask"]
    })

def judge_gate(gate, collected):
    raw = (JUDGE_PROMPT | llm | StrOutputParser()).invoke({
        "gate_rules": get_gate_rules(gate),
        "collected": collected
    })
    return safe_json(raw, {"satisfied": False, "reason": "Unable to validate"})


# =========================================================
# INTENT RESOLUTION
# =========================================================
def resolve_intent(q):
    docs = intent_retriever.invoke(q)
    return docs[0] if docs else None


# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="StartHub SOP Agent", layout="wide")
st.title("🧠 StartHub SOP Agent")

if "state" not in st.session_state:
    st.session_state.state = {
        "conversation_id": str(uuid.uuid4()),
        "intent_doc": None,
        "gate_order": [],
        "current_gate": None,
        "collected": {},
        "logs": []
    }

user_input = st.chat_input("Ask about your mail…")

if user_input:
    state = st.session_state.state
    now = datetime.utcnow().isoformat()

    # --- Intent ---
    if not state["intent_doc"]:
        intent_doc = resolve_intent(normalize(user_input))
        if not intent_doc:
            st.chat_message("assistant").markdown("I couldn’t find the right process.")
            st.stop()

        state["intent_doc"] = intent_doc
        state["gate_order"] = [
            g.strip() for g in intent_doc.metadata["gate_order"].split(">") if g.strip()
        ]
        state["current_gate"] = state["gate_order"][0]

    gate = state["current_gate"]
    state["collected"].setdefault(gate, "")

    result = judge_gate(gate, state["collected"][gate])

    # -------- LOG --------
    log = f"""
==============================
CONVERSATIONAL AGENT LOG (v1)
==============================

Turn ID:
{now}

Customer Message:
"{user_input}"

Intent:
- Selected: {state["intent_doc"].metadata["intent_id"]}

Gate Order:
{state["gate_order"]}

Current Gate:
{gate}

Data Read (expected):
{chr(10).join('- ' + d for d in get_required_data(gate))}

Collected Fields:
- {gate}: "{state["collected"][gate]}"

Gate Evaluation Summary:
{result["reason"]}

Gate Status:
- {gate} → {"PASSED" if result["satisfied"] else "PENDING"}
"""
    state["logs"].append(log)

    # --- Decision ---
    if result["satisfied"]:
        idx = state["gate_order"].index(gate)
        if idx + 1 < len(state["gate_order"]):
            state["current_gate"] = state["gate_order"][idx + 1]
            response = "Proceeding to the next step."
        else:
            response = "✅ All requirements met. Executing request."
    else:
        q = ask_gate(gate)
        combined = (state["collected"][gate] + " " + user_input).strip()
        if judge_gate(gate, combined)["satisfied"]:
            state["collected"][gate] = combined
        response = q

    st.chat_message("assistant").markdown(response)


# LOG VIEWER
with st.expander("📜 Full Conversational Logs"):
    for l in st.session_state.state["logs"]:
        st.code(l, language="text")
