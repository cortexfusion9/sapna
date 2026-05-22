import os
import json
import pandas as pd
import chromadb
import psycopg2  # Dynamic sync handler for native operations inside vector utility
from psycopg2.extras import RealDictCursor
from chromadb.utils import embedding_functions

# Define persistent storage path for ChromaDB
DB_PATH = "./chroma_db"
RULES_FILE = "intent_mapping_rules.xlsx" 

# 1. Initialize local persistent ChromaDB client and local embedding function
client = chromadb.PersistentClient(path=DB_PATH)
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

collection = client.get_or_create_collection(
    name="secure_sop_documents",
    embedding_function=embedding_fn
)


def get_db_connection():
    """Establishes a synchronized database connection for system verification tasks."""
    return psycopg2.connect(
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "Sapna123"),
        database=os.environ.get("DB_NAME", "gc_subsc"),
        host=os.environ.get("DB_HOST", "127.0.0.1"),
        port=int(os.environ.get("DB_PORT", 5432))
    )


def ingest_project_data():
    """Reads core intent guidelines from Excel into ChromaDB."""
    print("\n--- Starting Vector Database Ingestion (System Rules & Intents Only) ---")
    
    if not os.path.exists(RULES_FILE):
        print(f"[WARNING] {RULES_FILE} missing. Creating fallback default guidelines...")
        default_rules = {
            "intent_type": ["Check_Project_Status", "View_Project_Vault", "General_SOP_Query"],
            "discipline": ["General Contractor", "Electrical", "Plumbing"],
            "rule_description": [
                "Rule for routing project tracking, completion percentage, and baseline active milestones.",
                "Rule for handling secure credential tokens, costs payloads, financial records, and core secure keys.",
                "Standard operating guidelines for contractor interaction, system logic, and formatting layout constraints."
            ]
        }
        pd.DataFrame(default_rules).to_excel(RULES_FILE, index=False, engine="openpyxl")

    try:
        existing_data = collection.get()
        if existing_data and existing_data["ids"]:
            print(f"Clearing {len(existing_data['ids'])} existing vector documents...")
            collection.delete(ids=existing_data['ids'])
    except Exception as e:
        print(f"[INFO] Collection clear status: {str(e)}")

    documents = []
    metadatas = []
    ids = []

    try:
        df_rules = pd.read_excel(RULES_FILE, engine="openpyxl").fillna("")
        df_rules.columns = [c.strip().lower() for c in df_rules.columns]
        
        for index, row in df_rules.iterrows():
            doc_id = f"rule_{index}"
            intent_type = str(row.get("intent_type", "General_Query")).strip()
            discipline = str(row.get("discipline", "General Contractor")).strip()
            rule_desc = str(row.get("rule_description", "")).strip()

            doc_text = (
                f"Corporate Access Rule and Intent Guide - "
                f"Intent Context: {intent_type}, "
                f"Target Discipline Boundary: {discipline}, "
                f"Procedural Rule Description: {rule_desc}."
            )
            
            documents.append(doc_text)
            metadatas.append({
                "discipline": discipline,
                "intent_type": intent_type,
                "scope": "system_rules"
            })
            ids.append(doc_id)
            
        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            print(f"[SUCCESS] Vector DB Ingestion complete: {len(documents)} system rules vectorized.")
            return True
    except Exception as e:
        print(f"[ERROR] Ingesting System Rules failed: {str(e)}")
    return False


def build_rls_where_filter(role: str) -> dict:
    role_clean = role.strip()
    if role_clean.upper() in ["GC_OWNER", "GC OWNER", "ADMIN"]:
        return {"scope": "system_rules"}
    else:
        return {
            "$and": [
                {"discipline": role_clean},
                {"scope": "system_rules"}
            ]
        }


async def semantic_secure_search(user_id: str, user_query: str) -> str:
    """Verifies sessions directly using PostgreSQL instead of an Excel sheet."""
    role = None
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Verify user details natively from database
        cursor.execute(
            "SELECT user_role FROM public.gc_users WHERE LOWER(TRIM(user_id)) = LOWER(%s);", 
            (user_id.strip(),)
        )
        user_row = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if not user_row:
            return f"[PIPELINE ERROR] Access Denied: User '{user_id}' is unauthorized inside SQL Registry."
            
        role = user_row["user_role"]
        
    except Exception as db_err:
        return f"[PIPELINE ERROR] Session Auth Pipeline database connection failure: {str(db_err)}"

    where_filter = build_rls_where_filter(role)

    try:
        query_results = collection.query(
            query_texts=[user_query],
            n_results=1,
            where=where_filter
        )

        records = []
        if query_results and "documents" in query_results and query_results["documents"]:
            docs = query_results["documents"][0]
            metas = query_results["metadatas"][0] if "metadatas" in query_results else []

            for i, doc in enumerate(docs):
                meta = metas[i] if i < len(metas) else {}
                records.append({
                    "retrieved_context": doc,
                    "metadata": meta
                })

        if not records:
            return "[]"

        return json.dumps(records, indent=4)

    except Exception as e:
        return f"[PIPELINE ERROR] Semantic search failure: {str(e)}"