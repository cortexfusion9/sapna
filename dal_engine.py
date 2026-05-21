import os
import json
import pandas as pd
from setup_database import verify_user_session

# Define local Excel database file paths
STATUS_FILE = "gc_project_status.xlsx"
VAULT_FILE = "gc_project_vault.xlsx"

def secure_data_fetch(user_id: str, intent: str) -> list:
    """
    Secure Data Access Layer (DAL) function that fetches data from local Excel files.
    Enforces strict Row-Level Security (RLS) and Discipline filtering based on user permissions.
    """
    # 1. Verify user session and fetch permissions
    user_info = verify_user_session(user_id)
    if not user_info:
        return [{"error": f"Access Denied: Invalid or unauthorized user ID '{user_id}'."}]

    role, allowed_projects = user_info

    # 2. Determine target Excel file based on Intent routing
    if intent == "Check_Project_Status":
        target_file = STATUS_FILE
    elif intent == "View_Project_Vault":
        target_file = VAULT_FILE
    else:
        return [{"error": f"Access Denied: Invalid intent '{intent}' requested."}]

    # Ensure the target database file exists
    if not os.path.exists(target_file):
        return [{"error": f"Database file '{target_file}' not found."}]

    try:
        # 3. Load Excel database file into memory
        df = pd.read_excel(target_file, engine="openpyxl")

        # Validate that required schema columns are present
        if "project_id" not in df.columns or "discipline" not in df.columns:
            return [{"error": "Database schema mismatch: missing required RLS columns."}]

        # 4. ROW-LEVEL SECURITY (RLS) FILTERING

        # Filter A: Project Permissions (User must be explicitly allowed to view this project)
        allowed_projects_clean = [str(proj).upper().strip() for proj in allowed_projects]
        df_filtered = df[df["project_id"].astype(str).str.upper().str.strip().isin(allowed_projects_clean)]

        # Filter B: Discipline/Role Filtering
        role_clean = role.strip().upper()
        
        # Check if the user is a GC Owner/Admin (bypasses discipline filtering)
        if role_clean in ["GC_OWNER", "GC OWNER", "ADMIN"]:
            # Bypass discipline restrictions and return all project rows the admin is allowed to see
            pass
        elif role_clean in ["ELECTRICAL", "PLUMBING"]:
            # Enforce strict discipline matching (case-insensitive)
            df_filtered = df_filtered[df_filtered["discipline"].astype(str).str.upper().str.strip() == role_clean]
        else:
            # Deny access if user role is not supported or defined
            return [{"error": f"Access Denied: Role '{role}' does not have data access privileges."}]

        # 5. Return clean records list of dicts (JSON format)
        return df_filtered.to_dict(orient="records")

    except Exception as e:
        return [{"error": f"An unexpected error occurred during secure data fetch: {str(e)}"}]


if __name__ == "__main__":
    print("=" * 65)
    print("        RUNNING PHASE 2: SECURE DATA ACCESS LAYER TESTS")
    print("=" * 65)

    # Test Case 1: user_elec_456 asking for Check_Project_Status
    print("\n[TEST 1] user_elec_456 requesting Check_Project_Status:")
    print("Expected: ONLY Electrical status rows for allowed projects (PRJ001).")
    elec_status = secure_data_fetch("user_elec_456", "Check_Project_Status")
    print("Returned Result:")
    print(json.dumps(elec_status, indent=2))

    # Test Case 2: user_plumb_789 asking for View_Project_Vault
    print("\n[TEST 2] user_plumb_789 requesting View_Project_Vault:")
    print("Expected: ONLY Plumbing vault rows for allowed projects (PRJ001, PRJ002).")
    plumb_vault = secure_data_fetch("user_plumb_789", "View_Project_Vault")
    print("Returned Result:")
    print(json.dumps(plumb_vault, indent=2))

    # Test Case 3: Unauthorized Intent Request
    print("\n[TEST 3] user_elec_456 requesting unauthorized intent 'View_Confidential_Contracts':")
    print("Expected: Appropriate error block.")
    unauth_intent = secure_data_fetch("user_elec_456", "View_Confidential_Contracts")
    print("Returned Result:")
    print(json.dumps(unauth_intent, indent=2))

    # Test Case 4: Invalid/Fake User Request
    print("\n[TEST 4] Fake user_unknown_999 requesting Check_Project_Status:")
    print("Expected: Access Denied error block.")
    fake_user = secure_data_fetch("user_unknown_999", "Check_Project_Status")
    print("Returned Result:")
    print(json.dumps(fake_user, indent=2))

    print("\n" + "=" * 65)
