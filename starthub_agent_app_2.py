import streamlit as st
import os, re, json, uuid
import pandas as pd
from datetime import datetime
# from dotenv import load_dotenv   ← we remove this because Streamlit Cloud uses secrets

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =========================================================
# ENV (Streamlit Cloud version)
# =========================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing! Please add it in Streamlit Cloud → Secrets")
    st.stop()


# =========================================================
# CONFIG  (same as original)
# =========================================================
INTENT_XLSX = "Agent_Mapping_11.xlsx"
GATE_KNOWLEDGE_XLSX = "gate_knowledge.xlsx"
GATE_REQUIREMENTS_XLSX = "gate_requirements.xlsx"

INTENT_DB = "./chroma_intents"
GATE_DB = "./chroma_gates"

EMBEDDINGS = "text-embedding-3-large"
MODEL = "gpt-5.2"   # use real model in prod


# =========================================================
# All your original functions - NO CHANGES HERE
# =========================================================

def normalize(t):
    return re.sub(r"[^\w\s]", "", t.lower()).strip()

def safe_json(t, default):
    try:
        return json.loads(t)
    except:
        return default

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

        docs.append(
            Document(
                page_content=content.strip(),
                metadata={
                    "intent_id": str(r["Intent ID (input)"]),
                    "gate_order": str(r["GATE ORDER"]),
                    "action_flow": str(r["SUPPORT AGENT ACTION FLOW"])
                }
            )
        )
    return docs


def load_gate_knowledge():
    df = pd.read_excel(GATE_KNOWLEDGE_XLSX)
    gates = {}

    for _, r in df.iterrows():
        gates[str(r["Gate Name"]).strip()] = {
            "purpose": str(r["Purpose"]),
            "do_not_ask": str(r["Notes / Do Not Ask"])
        }
    return gates


def load_gate_requirements():
    df = pd.read_excel(GATE_REQUIREMENTS_XLSX)

    gates = {}

    for _, r in df.iterrows():
        gate = str(r["Gate Name"]).strip()

        gates.setdefault(gate, {
            "rules": [],
            "required_data": set()
        })

        content = f"""
GATE: {gate}

SATISFACTION PATHS:
{r['Satisfaction Paths']}

SATISFIED WHEN:
{r['Satisfied When']}
"""

        gates[gate]["rules"].append(
            Document(
                page_content=content.strip(),
                metadata={"gate": gate}
            )
        )

        if "Required Data" in r and not pd.isna(r["Required Data"]):
            for field in str(r["Required Data"]).split(";"):
                gates[gate]["required_data"].add(field.strip())

    return gates


# Vector stores ────────────────────────────────────────────────
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


def get_gate_rules(gate):
    docs = gate_retriever.invoke(gate)
    return docs[0].page_content if docs else "No rules found."

def get_required_data(gate):
    return sorted(gate_requirements.get(gate, {}).get("required_data", []))

llm = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY, temperature=0)


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


# =========================================================
#          STREAMLIT WRAPPER – minimal changes only
# =========================================================

st.set_page_config(page_title="StartHub SOP Agent", layout="wide")
st.title("🧠 StartHub SOP Agent")
st.caption("Exact same logic as console version – just with web chat interface")

# Session state – we simulate the same variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "turn" not in st.session_state:
    st.session_state.turn = 0
if "last_user_message" not in st.session_state:
    st.session_state.last_user_message = ""
if "intent_doc" not in st.session_state:
    st.session_state.intent_doc = None
if "collected" not in st.session_state:
    st.session_state.collected = {}
if "gate_order" not in st.session_state:
    st.session_state.gate_order = []
if "processing_intent" not in st.session_state:
    st.session_state.processing_intent = True   # like while not intent_doc


# Show conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
if user_input := st.chat_input("User:"):
    st.session_state.last_user_message = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if st.session_state.processing_intent:
            # === Intent detection phase ===
            intent_doc = resolve_intent(normalize(user_input))
            if not intent_doc:
                response = "Agent: I couldn’t find the right process. Please rephrase."
            else:
                st.session_state.intent_doc = intent_doc
                st.session_state.gate_order = [
                    g.strip() for g in intent_doc.metadata["gate_order"].split(">") if g.strip()
                ]
                st.session_state.processing_intent = False

                response = f"""
[DEBUG] Intent: {intent_doc.metadata['intent_id']}
[DEBUG] Gates: {" → ".join(st.session_state.gate_order)}
                """

                # Start first gate immediately
                first_gate = st.session_state.gate_order[0]
                st.session_state.collected.setdefault(first_gate, "")
                q = ask_gate(first_gate)

                response += f"""

{"="*80}
Current Gate: {first_gate}

Data Read (expected):
{chr(10).join(f"- {d}" for d in get_required_data(first_gate))}

Agent Message to Customer:
"{q}"
{"="*80}
                """

        else:
            # === Gate processing phase ===
            current_gate = st.session_state.gate_order[0]  # we process one by one, remove when done
            provided = st.session_state.collected[current_gate]
            st.session_state.turn += 1

            result = judge_gate(current_gate, provided)

            log = f"""
{"="*80}
Turn ID: {datetime.utcnow().isoformat()}
Customer Message:
"{st.session_state.last_user_message}"

Intent:
- Selected: {st.session_state.intent_doc.metadata['intent_id']}

Gate Order:
 → {" → ".join(st.session_state.gate_order)}

Current Gate:
{current_gate}

Data Read (expected):
{chr(10).join(f"- {d}" for d in get_required_data(current_gate))}

Collected Fields:
- {current_gate}: "{provided.strip() or '—'}"

Gate Evaluation Summary:
{result["reason"]}

Gate Status:
- {current_gate} → {'PASSED' if result['satisfied'] else 'PENDING'}

Agent Decision:
- Action: {"EXECUTED" if result["satisfied"] else "REQUESTED_MORE_INFORMATION"}
{"="*80}
"""

            if result["satisfied"]:
                st.session_state.collected[current_gate] = (provided + " " + user_input).strip()
                st.session_state.gate_order.pop(0)  # remove completed gate

                if not st.session_state.gate_order:
                    # All gates completed
                    summary = "\n".join(f"- {g}: \"{v.strip()}\"" for g,v in st.session_state.collected.items())
                    action = st.session_state.intent_doc.metadata["action_flow"]

                    log += f"""
{"═"*80}
✅ All gates satisfied

Collected Information Summary:
{summary}

🛠 SUPPORT AGENT ACTION FLOW:

{action}
{"═"*80}
"""
                    response = log
                else:
                    # Move to next gate
                    next_gate = st.session_state.gate_order[0]
                    st.session_state.collected.setdefault(next_gate, "")
                    q = ask_gate(next_gate)

                    response = log + f"""
Next Gate: {next_gate}

Agent Message to Customer:
"{q}"
{"="*80}
"""
            else:
                q = ask_gate(current_gate)
                response = log + f"""
Agent Message to Customer:
"{q}"
{"="*80}
"""

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


# Reset button
st.sidebar.title("Controls")
if st.sidebar.button("Start New Conversation"):
    st.session_state.clear()
    st.rerun()
