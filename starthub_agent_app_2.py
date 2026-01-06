from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import re
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =========================================================
# CONFIG
# =========================================================
PDF_PATH = "New Request Variations.pdf"
VECTOR_DB_DIR = "./chroma_support_agent"
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-5.2"


# =========================================================
# PDF → INTENT BLOCK PARSER (NO CHUNKING)
# =========================================================

def parse_intent_blocks(full_text: str):
    """
    Splits PDF text into intent-level documents.
    Assumes intent IDs are ALL CAPS with underscores.
    """
    blocks = re.split(r"\n(?=[A-Z_]{6,})", full_text)

    documents = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        intent = block.splitlines()[0].strip()

        documents.append(
            Document(
                page_content=block,
                metadata={"intent": intent}
            )
        )

    return documents


@st.cache_data
def load_documents():
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    full_text = "\n".join(p.page_content for p in pages)
    full_text = re.sub(r"\s+", " ", full_text)

    return parse_intent_blocks(full_text)


documents = load_documents()


# =========================================================
# VECTOR STORE
# =========================================================

@st.cache_resource
def build_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )


vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})


# =========================================================
# LLM
# =========================================================

llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)


# =========================================================
# PROMPTS
# =========================================================

RAG_PROMPT = PromptTemplate.from_template("""
You are a StartHub customer support agent.

STRICT RULES:
- Use ONLY the provided context
- DO NOT add, remove, or change steps
- DO NOT invent policies or checks
- The action flow MUST be copied verbatim

Context:
{context}

User Question:
{question}

Return EXACTLY in this format:

Agent Response:
<short helpful response>

StartHub Agent Action Flow:
<copy ALL numbered steps exactly>
""")

REACT_PROMPT = PromptTemplate.from_template("""
You are a StartHub support reasoning agent.

The user's request does not exactly match a request variation,
but the INTENT is similar.

Rules:
- Identify the closest intent
- Reuse the FULL action flow without modification
- Clearly state that this is inferred
- DO NOT invent steps

Context (known intents and workflows):
{context}

User Question:
{question}

Return EXACTLY in this format:

Agent Response:
<State this is based on closest matching intent>

StartHub Agent Action Flow:
<copy ALL numbered steps exactly>
""")


rag_chain = RAG_PROMPT | llm | StrOutputParser()
react_chain = REACT_PROMPT | llm | StrOutputParser()


# =========================================================
# HELPERS
# =========================================================

def has_numbered_steps(text: str) -> bool:
    return bool(re.search(r"\b1\.\s", text))


# =========================================================
# MAIN RESPONSE LOGIC (RAG + SAFE ReAct)
# =========================================================

def get_response(query: str) -> str:
    docs = retriever.invoke(query)

    if not docs:
        return "Sorry, I couldn’t find a matching or related process for your request."

    context = docs[0].page_content

    # Exact / strong match
    if has_numbered_steps(context):
        return rag_chain.invoke({
            "context": context,
            "question": query
        })

    # Safe ReAct fallback
    return react_chain.invoke({
        "context": context,
        "question": query
    })


# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(
    page_title="StartHub AI Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 StartHub AI Agent")
st.caption("Intent-aware | SOP-grounded | Safe ReAct fallback")

query = st.chat_input("Ask your request...")

if query:
    st.chat_message("user").write(query)

    with st.spinner("Analyzing request..."):
        response = get_response(query)

    st.chat_message("assistant").write(response)
