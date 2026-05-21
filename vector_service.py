import os
import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from setup_database import verify_user_session

# Define persistent storage path for ChromaDB
DB_PATH = "./chroma_db"
USERS_FILE = "users_gate_permissions.xlsx"
STATUS_FILE = "gc_project_status.xlsx"
VAULT_FILE = "gc_project_vault.xlsx"

# 1. Initialize local persistent ChromaDB client and local embedding function
client = chromadb.PersistentClient(path=DB_PATH)

# Use Chroma's default lightweight local embedding function (SentenceTransformer MiniLM)
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

# Get or create the secure SOP collection
collection = client.get_or_create_collection(
    name="secure_sop_documents",
    embedding_function=embedding_fn
)


def ingest_project_data():
    """
    Reads the local project status and project vault Excel files,
    constructs text representations for each row, and ingests them
    into ChromaDB with RLS-compliant metadata tags.
    """
    print("\n--- Starting Vector Database Ingestion ---")
    
    # Verify Excel database files exist
    if not os.path.exists(STATUS_FILE) or not os.path.exists(VAULT_FILE):
        print("[ERROR] Excel files missing. Please run setup_database() first.")
        return False

    # Clear existing collection items before fresh ingestion to avoid duplicates
    try:
        existing_data = collection.get()
        if existing_data and existing_data["ids"]:
            print(f"Clearing {len(existing_data['ids'])} existing vector documents...")
            collection.delete(ids=existing_data["ids"])
    except Exception as e:
        print(f"[INFO] Collection clear status: {str(e)}")

    documents = []
    metadatas = []
    ids = []

    # 1. Ingest Status Data (gc_project_status.xlsx)
    try:
        df_status = pd.read_excel(STATUS_FILE, engine="openpyxl")
        print(f"Reading {STATUS_FILE} (found {len(df_status)} records)...")
        for index, row in df_status.iterrows():
            doc_id = f"status_{row.get('id', index)}"
            proj_id = str(row.get("project_id", "")).strip()
            proj_name = str(row.get("project_name", "")).strip()
            discipline = str(row.get("discipline", "")).strip()
            status = str(row.get("current_status", "")).strip()
            last_updated = str(row.get("last_updated", "")).strip()

            # Represent document text for semantic search
            doc_text = (
                f"Project Status Details - "
                f"Project Name: {proj_name}, "
                f"Project ID: {proj_id}, "
                f"Discipline: {discipline}, "
                f"Current Status: {status}, "
                f"Last Updated: {last_updated}."
            )
            
            documents.append(doc_text)
            metadatas.append({
                "project_id": proj_id,
                "project_name": proj_name,
                "discipline": discipline,
                "intent_type": "Check_Project_Status"
            })
            ids.append(doc_id)
    except Exception as e:
        print(f"[ERROR] Ingesting Status sheet failed: {str(e)}")
        return False

    # 2. Ingest Vault Data (gc_project_vault.xlsx)
    try:
        df_vault = pd.read_excel(VAULT_FILE, engine="openpyxl")
        print(f"Reading {VAULT_FILE} (found {len(df_vault)} records)...")
        for index, row in df_vault.iterrows():
            doc_id = f"vault_{row.get('id', index)}"
            proj_id = str(row.get("project_id", "")).strip()
            proj_name = str(row.get("project_name", "")).strip()
            discipline = str(row.get("discipline", "")).strip()
            cost = str(row.get("total_project_cost", "")).strip()
            payload = str(row.get("secure_data_payload", "")).strip()

            # Represent document text for semantic search
            doc_text = (
                f"Project Vault Confidential Details - "
                f"Project Name: {proj_name}, "
                f"Project ID: {proj_id}, "
                f"Discipline: {discipline}, "
                f"Total Project Cost: ${cost}, "
                f"Secure Data Payload: {payload}."
            )
            
            documents.append(doc_text)
            metadatas.append({
                "project_id": proj_id,
                "project_name": proj_name,
                "discipline": discipline,
                "intent_type": "View_Project_Vault"
            })
            ids.append(doc_id)
    except Exception as e:
        print(f"[ERROR] Ingesting Vault sheet failed: {str(e)}")
        return False

    # Perform bulk ingestion to ChromaDB
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"[SUCCESS] Vector DB Ingestion complete: {len(documents)} documents vectorized.")
        return True
    
    return False


def build_rls_where_filter(role: str, allowed_projects: list[str]) -> dict:
    """
    Helper function that builds a strict ChromaDB metadata 'where' filter
    to enforce RLS (Row-Level Security) at the vector DB query layer.
    """
    role_clean = role.strip()
    
    # Build project constraint filter
    if len(allowed_projects) == 1:
        project_filter = {"project_id": allowed_projects[0]}
    else:
        project_filter = {"project_id": {"$in": allowed_projects}}

    # GC Admin/Owner bypasses discipline constraints
    if role_clean.upper() in ["GC_OWNER", "GC OWNER", "ADMIN"]:
        return project_filter
    else:
        # Enforce both discipline match and allowed projects match
        return {
            "$and": [
                {"discipline": role_clean},
                project_filter
            ]
        }


def semantic_secure_search(user_id: str, user_query: str) -> str:
    """
    Core semantic retrieval function.
    Verifies user, builds strict RLS query filters, performs local semantic search via ChromaDB,
    and returns a structured JSON string of RLS-compliant matching records.
    """
    # 1. Look up user session from permissions Excel sheet
    user_info = verify_user_session(user_id)
    if not user_info:
        return f"[PIPELINE ERROR] Access Denied: User '{user_id}' is unauthorized."

    role, allowed_projects = user_info

    # 2. Build RLS-compliant ChromaDB metadata query filters
    where_filter = build_rls_where_filter(role, allowed_projects)

    try:
        # 3. Perform semantic query inside local persistent ChromaDB collection
        query_results = collection.query(
            query_texts=[user_query],
            n_results=2,  # Return top matches
            where=where_filter
        )

        # 4. Format RLS-secured documents into a structured response
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
            return "[]"  # No matching authorized data found

        return json.dumps(records, indent=4)

    except Exception as e:
        return f"[PIPELINE ERROR] Semantic search failure: {str(e)}"


if __name__ == "__main__":
    # Automatic test suite
    print("=" * 70)
    print("       RUNNING VECTOR DATABASE INGESTION & SECURE SEMANTIC RAG TESTS")
    print("=" * 70)

    # 1. Ingest Excel spreadsheets into ChromaDB vectors
    ingest_project_data()

    # 2. Test semantic search for user_elec_456 (Electrical Subcontractor, allowed PRJ001)
    print("\n[TEST 1] user_elec_456 asking semantically: 'Show me status and updates for Apex Tower'")
    res1 = semantic_secure_search("user_elec_456", "Show me status and updates for Apex Tower")
    print("RLS Context Returned:")
    print(res1)

    # 3. Test semantic search for user_plumb_789 (Plumbing Subcontractor, allowed PRJ001, PRJ002)
    print("\n[TEST 2] user_plumb_789 asking semantically: 'What are the secure costs payload of my projects?'")
    res2 = semantic_secure_search("user_plumb_789", "What are the secure costs payload of my projects?")
    print("RLS Context Returned:")
    print(res2)

    # 4. Verification test: Checking if Electrical user can retrieve Plumbing details (Security Check)
    print("\n[TEST 3 - SECURITY AUDIT] user_elec_456 trying to search plumbing details semantically:")
    res3 = semantic_secure_search("user_elec_456", "Show me plumbing layout hashes and cost data")
    print("RLS Context Returned:")
    print(res3)

    print("\n" + "=" * 70)
