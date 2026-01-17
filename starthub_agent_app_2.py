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
    st.error("OPENAI_API_KEY is missing!")
    st.stop()

INTENT_XLSX = "Agent_Mapping_11.xlsx"
GATE_KNOWLEDGE_XLSX = "gate_knowledge.xlsx"
GATE_REQUIREMENTS_XLSX = "gate_requirements.xlsx"

INTENT_DB = "./chroma_intents"
GATE_DB = "./chroma_gates"

EMBEDDINGS_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"  # ← change to your real model

# ────────────────────────────────────────────────
#                   UTILITIES
# ────────────────────────────────────────────────
def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

def safe_json(t, default):
    try:
        return json.loads(t)
    except:
        return default

# ────────────────────────────────────────────────
#              DATA LOADING & VECTOR STORES
# ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge bases...")
def load_all_data():
    embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL, api_key=OPENAI_API_KEY)

    # Intents
    df = pd.read_excel(INTENT_XLSX)
    intent_docs = []
    for _, r in df.iterrows():
        if pd.isna(r["SUPPORT AGENT ACTION FLOW"]):
            continue
        content = f"""
INTENT:
{r['Intent ID (input)']}

REQUEST_VARIATIONS:
{r['Request Variations']}
""".strip()
        intent_docs.append(Document(
            page_content=content,
            metadata={
                "intent_id": str(r["Intent ID (input)"]),
                "gate_order": str(r["GATE ORDER"]),
                "action_flow": str(r["SUPPORT AGENT ACTION FLOW"])
            }
        ))

    intent_vs = Chroma.from_documents(intent_docs, embeddings, persist_directory=INTENT_DB)
    intent_retriever = intent_vs.as_retriever(search_kwargs={"k": 1})

    # Gate knowledge
    df_k = pd.read_excel(GATE_KNOWLEDGE_XLSX)
    gate_knowledge = {}
    for _, r in df_k.iterrows():
        name = str(r["Gate Name"]).strip()
        gate_knowledge[name] = {
            "purpose": str(r["Purpose"]),
            "do_not_ask": str(r["Notes / Do Not Ask"])
        }

    # Gate requirements
    df_req = pd.read_excel(GATE_REQUIREMENTS_XLSX)
    gate_requirements = {}
    gate_docs = []
    for _, r in df_req.iterrows():
        gate = str(r["Gate Name"]).strip()
        gate_requirements.setdefault(gate, {"rules": [], "required_data": set()})

        content = f"""
GATE: {gate}

SATISFACTION PATHS:
{r['Satisfaction Paths']}

SATISFIED WHEN:
{r['Satisfied When']}
""".strip()

        gate_docs.append(Document(page_content=content, metadata={"gate": gate}))

        if "Required Data" in r and pd.notna(r["Required Data"]):
            for field in str(r["Required Data"]).split(";"):
                gate_requirements[gate]["required_data"].add(field.strip())

    gate_vs = Chroma.from_documents(gate_docs, embeddings, persist_directory=GATE_DB)
    gate_retriever = gate_vs.as_retriever(search_kwargs={"k": 1})

    return intent_retriever, gate_retriever, gate_knowledge, gate_requirements


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

# ────────────────────────────────────────────────
#                    HELPERS
# ────────────────────────────────────────────────
def get_gate_rules(gate):
    docs = gate_retriever.invoke(gate)
    return docs[0].page_content if docs else "No rules found."

def get_required_data(gate):
    return sorted(gate_requirements.get(gate, {}).get("required_data", []))

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

def resolve_intent(q):
    docs = intent_retriever.invoke(q)
    return docs[0] if docs else None

# ────────────────────────────────────────────────
#               STREAMLIT APP
# ────────────────────────────────────────────────
st.set_page_config(page_title="StartHub SOP Agent - Detailed Logs", layout="wide")

st.title("🧠 StartHub SOP Agent  ·  Detailed Log Mode")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "turn" not in st.session_state:
    st.session_state.turn = 0
if "intent_doc" not in st.session_state:
    st.session_state.intent_doc = None
if "collected" not in st.session_state:
    st.session_state.collected = {}
if "gate_order" not in st.session_state:
    st.session_state.gate_order = []
if "current_gate_idx" not in st.session_state:
    st.session_state.current_gate_idx = 0

# Display chat history (we keep it for context)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("User:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    last_user_message = prompt.strip()

    with st.chat_message("assistant"):
        placeholder = st.empty()

        if st.session_state.intent_doc is None:
            # ── Intent detection ───────────────────────────────
            intent_doc = resolve_intent(normalize(last_user_message))
            if intent_doc is None:
                response = "Agent: I couldn’t find the right process. Please rephrase."
            else:
                st.session_state.intent_doc = intent_doc
                st.session_state.gate_order = [
                    g.strip() for g in intent_doc.metadata["gate_order"].split(">") if g.strip()
                ]
                st.session_state.current_gate_idx = 0

                response = f"""
**Intent selected:** {intent_doc.metadata['intent_id']}  
**Gates:** {' → '.join(st.session_state.gate_order)}
                """

        else:
            # ── Gate processing ────────────────────────────────
            idx = st.session_state.current_gate_idx
            if idx >= len(st.session_state.gate_order):
                # All done
                summary = "\n".join(
                    f"- {g}: \"{v.strip()}\"" for g, v in st.session_state.collected.items()
                )
                action = st.session_state.intent_doc.metadata["action_flow"]

                response = f"""
{"═"*80}
✅ All gates satisfied

Collected Information Summary:
{summary}

🛠 SUPPORT AGENT ACTION FLOW:

{action}
{"═"*80}
                """
            else:
                gate = st.session_state.gate_order[idx]
                st.session_state.collected.setdefault(gate, "")

                provided = st.session_state.collected[gate]
                result = judge_gate(gate, provided + " " + last_user_message)

                st.session_state.turn += 1

                log_block = f"""
{"="*80}
Turn ID: {datetime.utcnow().isoformat()}
Customer Message:
"{last_user_message}"

Intent:
- Selected: {st.session_state.intent_doc.metadata['intent_id']}

Gate Order:
 → {' → '.join(st.session_state.gate_order)}

Current Gate:
{gate}

Data Read (expected):
{chr(10).join(f"- {d}" for d in get_required_data(gate))}

Collected Fields:
- {gate}: "{provided.strip()}"

Gate Evaluation Summary:
{result["reason"]}

Gate Status:
- {gate} → {'PASSED' if result["satisfied"] else 'PENDING'}

Agent Decision:
- Action: {"EXECUTED" if result["satisfied"] else "REQUESTED_MORE_INFORMATION"}
{"="*80}
"""

                if result["satisfied"]:
                    st.session_state.collected[gate] = (provided + " " + last_user_message).strip()
                    st.session_state.current_gate_idx += 1
                    response = log_block + "\nGate passed → continuing..."
                else:
                    q = ask_gate(gate)
                    response = log_block + f"""
Agent Message to Customer:
"{q}"
{"="*80}
"""

                    # Show question to user
                    st.markdown(f"**Agent:** {q}")

        placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Reset button
if st.button("New Conversation", type="primary"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
