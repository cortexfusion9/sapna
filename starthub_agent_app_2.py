import streamlit as st
import os
import re
import json
import uuid
import pandas as pd
from datetime import datetime

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ────────────────────────────────────────────────
#                 CONFIG & ENV
# ────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY environment variable is missing!")
    st.stop()

INTENT_XLSX = "Agent_Mapping_11.xlsx"
GATE_KNOWLEDGE_XLSX = "gate_knowledge.xlsx"
GATE_REQUIREMENTS_XLSX = "gate_requirements.xlsx"

INTENT_DB = "./chroma_intents"
GATE_DB = "./chroma_gates"

EMBEDDINGS_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-5.2"          # ← change to your desired model (gpt-5.2 → probably typo)

# ────────────────────────────────────────────────
#                   UTILITIES
# ────────────────────────────────────────────────
def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

def safe_json_parse(text: str, default=None):
    try:
        return json.loads(text)
    except:
        return default

# ────────────────────────────────────────────────
#              DATA LOADING & VECTOR STORES
# ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge bases...")
def load_all_data():
    embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL, api_key=OPENAI_API_KEY)

    # Intents
    df_intent = pd.read_excel(INTENT_XLSX)
    intent_docs = []
    for _, row in df_intent.iterrows():
        if pd.isna(row.get("SUPPORT AGENT ACTION FLOW")):
            continue
        content = f"INTENT:\n{row['Intent ID (input)']}\n\nREQUEST_VARIATIONS:\n{row['Request Variations']}"
        intent_docs.append(Document(
            page_content=content.strip(),
            metadata={
                "intent_id": str(row["Intent ID (input)"]),
                "gate_order": str(row["GATE ORDER"]),
                "action_flow": str(row["SUPPORT AGENT ACTION FLOW"])
            }
        ))

    intent_vs = Chroma.from_documents(intent_docs, embeddings, persist_directory=INTENT_DB)
    intent_retriever = intent_vs.as_retriever(search_kwargs={"k": 1})

    # Gates knowledge
    df_knowledge = pd.read_excel(GATE_KNOWLEDGE_XLSX)
    gate_knowledge = {}
    for _, row in df_knowledge.iterrows():
        gate_name = str(row["Gate Name"]).strip()
        gate_knowledge[gate_name] = {
            "purpose": str(row["Purpose"]),
            "do_not_ask": str(row["Notes / Do Not Ask"])
        }

    # Gate requirements & rules
    df_req = pd.read_excel(GATE_REQUIREMENTS_XLSX)
    gate_requirements = {}
    gate_rule_docs = []

    for _, row in df_req.iterrows():
        gate = str(row["Gate Name"]).strip()
        gate_requirements.setdefault(gate, {"rules": [], "required_data": set()})

        content = f"""
GATE: {gate}

SATISFACTION PATHS:
{row['Satisfaction Paths']}

SATISFIED WHEN:
{row['Satisfied When']}
""".strip()

        gate_rule_docs.append(Document(page_content=content, metadata={"gate": gate}))

        if "Required Data" in row and pd.notna(row["Required Data"]):
            for field in str(row["Required Data"]).split(";"):
                gate_requirements[gate]["required_data"].add(field.strip())

    gate_vs = Chroma.from_documents(gate_rule_docs, embeddings, persist_directory=GATE_DB)
    gate_retriever = gate_vs.as_retriever(search_kwargs={"k": 1})

    return intent_retriever, gate_retriever, gate_knowledge, gate_requirements


# Load everything once
intent_retriever, gate_retriever, gate_knowledge, gate_requirements = load_all_data()

# ────────────────────────────────────────────────
#                    LLM & PROMPTS
# ────────────────────────────────────────────────
llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY, temperature=0)

ASK_PROMPT = PromptTemplate.from_template("""
You are a professional customer support agent.

GATE RULES:
{gate_rules}

Purpose:
{purpose}

Do NOT ask:
{do_not_ask}

Ask **the single best question** to satisfy this gate.
Return **only the question** — nothing else.
""")

JUDGE_PROMPT = PromptTemplate.from_template("""
You are a strict gate validation engine.

GATE RULES:
{gate_rules}

User has provided so far for this gate:
{collected}

Decide if the gate is **satisfied**.

Return JSON only:
{{
  "satisfied": true/false,
  "reason": "short explanation"
}}
""")


def get_gate_rules(gate: str) -> str:
    docs = gate_retriever.invoke(gate)
    return docs[0].page_content if docs else "No rules found."


def get_required_data(gate: str) -> list:
    return sorted(gate_requirements.get(gate, {}).get("required_data", []))


def ask_for_gate(gate: str) -> str:
    return (ASK_PROMPT | llm | StrOutputParser()).invoke({
        "gate_rules": get_gate_rules(gate),
        "purpose": gate_knowledge.get(gate, {}).get("purpose", ""),
        "do_not_ask": gate_knowledge.get(gate, {}).get("do_not_ask", "")
    })


def judge_gate(gate: str, collected: str) -> dict:
    raw = (JUDGE_PROMPT | llm | StrOutputParser()).invoke({
        "gate_rules": get_gate_rules(gate),
        "collected": collected
    })
    return safe_json_parse(raw, {"satisfied": False, "reason": "Parse error"})


def resolve_intent(query: str) -> Document | None:
    docs = intent_retriever.invoke(normalize(query))
    return docs[0] if docs else None


# ────────────────────────────────────────────────
#               STREAMLIT INTERFACE
# ────────────────────────────────────────────────
st.set_page_config(page_title="StartHub SOP Agent", layout="wide")

st.title("🧠 StartHub SOP Agent")
st.caption("Conversational support agent following defined gates & flows")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "intent_doc" not in st.session_state:
    st.session_state.intent_doc = None
if "collected" not in st.session_state:
    st.session_state.collected = {}
if "gate_order" not in st.session_state:
    st.session_state.gate_order = []
if "current_gate_idx" not in st.session_state:
    st.session_state.current_gate_idx = 0
if "all_gates_done" not in st.session_state:
    st.session_state.all_gates_done = False

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.all_gates_done:
            st.info("Conversation completed. Start new conversation if needed.")
        elif st.session_state.intent_doc is None:
            # First phase → detect intent
            intent_doc = resolve_intent(prompt)
            if intent_doc is None:
                response = "Sorry, I couldn't match your request to any known process.\nPlease try to rephrase."
            else:
                st.session_state.intent_doc = intent_doc
                st.session_state.gate_order = [
                    g.strip() for g in intent_doc.metadata["gate_order"].split(">") if g.strip()
                ]
                st.session_state.current_gate_idx = 0

                response = f"""
**Intent detected:** {intent_doc.metadata['intent_id']}

**Gate sequence:** {' → '.join(st.session_state.gate_order)}
                """
        else:
            # Gate phase
            if st.session_state.current_gate_idx >= len(st.session_state.gate_order):
                st.session_state.all_gates_done = True
                action = st.session_state.intent_doc.metadata["action_flow"]
                response = f"**All gates satisfied!**\n\n**Action flow to execute:**\n\n{action}"
            else:
                gate = st.session_state.gate_order[st.session_state.current_gate_idx]
                st.session_state.collected.setdefault(gate, "")

                current_collected = st.session_state.collected[gate]
                judge_result = judge_gate(gate, current_collected + " " + prompt)

                debug_content = f"""
**Current Gate:** {gate}  
**Required fields:** {', '.join(get_required_data(gate)) or '—'}  
**Collected so far:** {current_collected.strip() or '—'}  
**Evaluation:** {judge_result['reason']}  
**Status:** {'✅ PASSED' if judge_result['satisfied'] else '⏳ PENDING'}
                """

                if judge_result["satisfied"]:
                    st.session_state.collected[gate] = (current_collected + " " + prompt).strip()
                    st.session_state.current_gate_idx += 1

                    if st.session_state.current_gate_idx >= len(st.session_state.gate_order):
                        st.session_state.all_gates_done = True
                        action = st.session_state.intent_doc.metadata["action_flow"]
                        response = f"{debug_content}\n\n**ALL GATES COMPLETED!**\n\n**Final Action Flow:**\n{action}"
                    else:
                        response = f"{debug_content}\n\n**Gate passed → moving forward...**"
                else:
                    question = ask_for_gate(gate)
                    response = f"{debug_content}\n\n**Question for you:**\n> {question}"

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# ────────────────────────────────────────────────
#                   SIDEBAR CONTROLS
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    if st.button("🔄 New Conversation", type="primary"):
        for key in list(st.session_state.keys()):
            if key != "messages":  # keep messages for history view if you want
                del st.session_state[key]
        st.session_state.messages = []
        st.rerun()

    st.caption("OpenAI model: " + LLM_MODEL)
    st.caption(f"Session: {uuid.uuid4().hex[:8]}")
