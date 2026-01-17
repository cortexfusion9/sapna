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
MODEL = "gpt-4o-mini"   # ✅ real production-safe model

# =========================================================
# STREAMLIT STATE (CRITICAL FIX)
# =========================================================
if "state" not in st.session_state:
    st.session_state.state = {
        "conversation_id": str(uuid.uuid4()),
        "messages": [],
        "logs": [],
        "intent_doc": None,
        "gate_order": [],
        "collected": {},
        "current_gate": None,
        "last_user_message": ""
    }

state = st.session_state.state

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
@st.cache_resource
def load_intents():
    df = pd.read_excel(INTENT_XLSX)
    docs = []
    for _, r in df.iterrows():
        if pd.isna(r["SUPPORT AGENT ACTION FLOW"]):
            continue
        content = f"""
INTENT:
{r['Intent ID (input)']}

REQUEST_VARIATIONS:
{r['Request Variations']}
"""
        docs.append(Document(
            page_content=content.strip(),
            metadata={
                "intent_id": r["Intent ID (input)"],
                "gate_order": r["GATE ORDER"],
                "action_flow": r["SUPPORT AGENT ACTION FLOW"]
            }
        ))
    return docs

@st.cache_resource
def load_gate_knowledge():
    df = pd.read_excel(GATE_KNOWLEDGE_XLSX)
    return {
        r["Gate Name"]: {
            "purpose": r["Purpose"],
            "do_not_ask": r["Notes / Do Not Ask"]
        }
        for _, r in df.iterrows()
    }

@st.cache_resource
def load_gate_requirements():
    df = pd.read_excel(GATE_REQUIREMENTS_XLSX)
    gates = {}
    for _, r in df.iterrows():
        gate = r["Gate Name"]
        gates.setdefault(gate, {
            "rules": [],
            "required_data": []
        })
        gates[gate]["rules"].append(
            f"""
SATISFACTION PATHS:
{r['Satisfaction Paths']}

SATISFIED WHEN:
{r['Satisfied When']}
"""
        )
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
You are a customer support agent.

Gate Rules:
{rules}

Purpose:
{purpose}

Do NOT ask:
{do_not_ask}

Ask ONE clear question to satisfy the gate.
Return only the question.
""")

JUDGE_PROMPT = PromptTemplate.from_template("""
Gate Rules:
{rules}

User Provided:
{input}

Is the gate satisfied?

Return JSON:
{{"satisfied": true/false, "reason": "..."}}
""")

# =========================================================
# GATE HELPERS
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
def log_turn(*, customer_message, gate, result):
    log = {
        "timestamp": datetime.utcnow().isoformat(),
        "conversation_id": state["conversation_id"],
        "customer_message": customer_message,
        "intent": state["intent_doc"].metadata["intent_id"],
        "gate": gate,
        "gate_status": "PASSED" if result["satisfied"] else "PENDING",
        "data_read": gate_requirements.get(gate, {}).get("required_data", []),
        "collected": state["collected"].get(gate, ""),
        "reason": result["reason"]
    }
    state["logs"].append(log)

# =========================================================
# UI
# =========================================================
st.title("🧠 StartHub SOP Agent")

user_input = st.chat_input("Type your request")

if user_input:
    state["messages"].append(("user", user_input))
    state["last_user_message"] = user_input

    # ---- Intent resolution ----
    if not state["intent_doc"]:
        state["intent_doc"] = intent_retriever.invoke(normalize(user_input))[0]
        state["gate_order"] = [
            g.strip()
            for g in state["intent_doc"].metadata["gate_order"].split(">")
        ]
        for g in state["gate_order"]:
            state["collected"][g] = ""

    # ---- Gate execution ----
    for gate in state["gate_order"]:
        provided = state["collected"][gate]
        result = judge_gate(gate, provided)
        log_turn(customer_message=user_input, gate=gate, result=result)

        if not result["satisfied"]:
            question = ask_gate(gate)
            state["messages"].append(("assistant", question))
            st.chat_message("assistant").write(question)
            break
        else:
            continue
    else:
        final_msg = "✅ All requirements satisfied. Executing request."
        state["messages"].append(("assistant", final_msg))
        st.chat_message("assistant").write(final_msg)

# =========================================================
# CHAT HISTORY
# =========================================================
for role, msg in state["messages"]:
    st.chat_message(role).write(msg)

# =========================================================
# LOG VIEW (DEBUG PANEL)
# =========================================================
with st.expander("📜 Internal Logs"):
    st.json(state["logs"])
