import json
from dal_engine import secure_data_fetch

def detect_intent(user_query: str) -> str:
    """
    Keyword-based mock intent classifier.
    Maps queries to 'Check_Project_Status' or 'View_Project_Vault' based on keywords.
    """
    query_lower = user_query.lower()
    
    # Intent 1: Check_Project_Status keywords
    status_keywords = ["status", "progress", "update"]
    if any(kw in query_lower for kw in status_keywords):
        return "Check_Project_Status"
        
    # Intent 2: View_Project_Vault keywords
    vault_keywords = ["vault", "cost", "budget", "payload"]
    if any(kw in query_lower for kw in vault_keywords):
        return "View_Project_Vault"
        
    return "Unknown"

def run_agent_pipeline(user_id: str, user_query: str) -> str:
    """
    Core orchestrator pipeline that detects intent, fetches RLS-secured data,
    and injects it into a structured prompt context ready for the LLM.
    """
    # 1. Detect Intent
    intent = detect_intent(user_query)
    if intent == "Unknown":
        return f"[PIPELINE ERROR] Access Denied: Query intent could not be mapped to any authorized action."

    # 2. Fetch secured data from DAL
    data_records = secure_data_fetch(user_id, intent)
    
    # 3. Check if DAL returned an error
    if len(data_records) == 1 and "error" in data_records[0]:
        return f"[PIPELINE ERROR] {data_records[0]['error']}"
        
    # 4. Construct Context and Prompt Injection
    context_str = json.dumps(data_records, indent=4)
    
    prompt_template = f"""================================================================================
SYSTEM PROMPT (Secure SOP Agent Context)
================================================================================
You are an expert Secure SOP Agent.
Based ONLY on the verified corporate database records provided below, answer the user's query.
If the information is not present in the data, state that you do not have authorization or the data is not available.

[VERIFIED DATA CONTEXT]:
{context_str}

--------------------------------------------------------------------------------
USER QUERY: {user_query}
--------------------------------------------------------------------------------
================================================================================"""

    return prompt_template


if __name__ == "__main__":
    print("=" * 75)
    print("        RUNNING PHASE 3: AGENT CORE ORCHESTRATION & PIPELINE TESTS")
    print("=" * 75)

    # Test Case A: user_elec_456 asking for progress and status
    print("\n[TEST A] user_elec_456 asking: 'What is the progress and status of Apex Tower?'")
    pipeline_res_a = run_agent_pipeline(
        user_id="user_elec_456", 
        user_query="What is the progress and status of Apex Tower?"
    )
    print(pipeline_res_a)

    # Test Case B: user_plumb_789 asking for secure vault costs
    print("\n[TEST B] user_plumb_789 asking: 'Can I see the secure vault costs for my piping projects?'")
    pipeline_res_b = run_agent_pipeline(
        user_id="user_plumb_789", 
        user_query="Can I see the secure vault costs for my piping projects?"
    )
    print(pipeline_res_b)

    # Test Case C: Intent cannot be mapped clearly
    print("\n[TEST C] user_elec_456 asking: 'What is the weather like today?'")
    pipeline_res_c = run_agent_pipeline(
        user_id="user_elec_456", 
        user_query="What is the weather like today?"
    )
    print(pipeline_res_c)

    print("\n" + "=" * 75)
