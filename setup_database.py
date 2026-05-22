import os
import pandas as pd

# Define Excel file paths in the local directory
USERS_FILE = "users_gate_permissions.xlsx"
STATUS_FILE = "gc_project_status.xlsx"
VAULT_FILE = "gc_project_vault.xlsx"

def setup_database():
    """
    Checks if the local Excel database files exist.
    If not, creates them with modern mock data structure for secure SOP verification.
    """
    print("Checking local Excel files status...")

    # 1. Create users_gate_permissions.xlsx
    if not os.path.exists(USERS_FILE):
        print(f"Creating {USERS_FILE}...")
        users_data = {
            "user_id": ["user_elec_456", "user_plumb_789", "gc_admin_100"],
            "user_name": ["John Electrician", "David Plumber", "Alice Admin"],
            "user_role": ["Electrical", "Plumbing", "GC_Owner"],
            "allowed_projects": ["PRJ001", "PRJ001,PRJ002", "PRJ001,PRJ002,PRJ003"]
        }
        df_users = pd.DataFrame(users_data)
        df_users.to_excel(USERS_FILE, index=False, engine="openpyxl")
        print(f"[SUCCESS] {USERS_FILE} created successfully.")
    else:
        print(f"[INFO] {USERS_FILE} already exists.")

    # 2. Create gc_project_status.xlsx
    if not os.path.exists(STATUS_FILE):
        print(f"Creating {STATUS_FILE}...")
        status_data = {
            "id": [1, 2, 3],
            "project_id": ["PRJ001", "PRJ001", "PRJ002"],
            "project_name": ["Apex Tower", "Apex Tower", "Vertex Plaza"],
            "discipline": ["Electrical", "Plumbing", "Electrical"],
            "current_status": ["In Progress", "Completed", "Pending Approval"],
            "last_updated": ["2026-05-20", "2026-05-18", "2026-05-21"]
        }
        df_status = pd.DataFrame(status_data)
        df_status.to_excel(STATUS_FILE, index=False, engine="openpyxl")
        print(f"[SUCCESS] {STATUS_FILE} created successfully.")
    else:
        print(f"[INFO] {STATUS_FILE} already exists.")

    # 3. Create gc_project_vault.xlsx
    if not os.path.exists(VAULT_FILE):
        print(f"Creating {VAULT_FILE}...")
        vault_data = {
            "id": [1, 2, 3],
            "project_id": ["PRJ001", "PRJ001", "PRJ002"],
            "project_name": ["Apex Tower", "Apex Tower", "Vertex Plaza"],
            "discipline": ["Electrical", "Plumbing", "Electrical"],
            "total_project_cost": [120000, 85000, 450000],
            "secure_data_payload": [
                '{"wiring_diagram_hash": "a8f93b", "inspection_status": "passed"}',
                '{"piping_layout_hash": "c7e2b1", "inspection_status": "passed"}',
                '{"substation_specs": "high_voltage_33kv", "risk_level": "medium"}'
            ]
        }
        df_vault = pd.DataFrame(vault_data)
        df_vault.to_excel(VAULT_FILE, index=False, engine="openpyxl")
        print(f"[SUCCESS] {VAULT_FILE} created successfully.")
    else:
        print(f"[INFO] {VAULT_FILE} already exists.")

    print("\nAll database setup checks completed successfully!")


def verify_user_session(user_id: str) -> tuple[str, list[str]] | None:
    """
    Reads users_gate_permissions.xlsx and verifies the given user_id.
    Returns a tuple of (user_role, list of allowed_projects) or None if not found/invalid.
    """
    try:
        # Check if the permissions file exists
        if not os.path.exists(USERS_FILE):
            print(f"[ERROR] Permission file '{USERS_FILE}' does not exist. Run setup_database() first.")
            return None

        # Read the Excel sheet using pandas
        df = pd.read_excel(USERS_FILE, engine="openpyxl")

        # Validate required columns are present
        required_cols = ["user_id", "user_role", "allowed_projects"]
        if not all(col in df.columns for col in required_cols):
            print(f"[ERROR] Required columns {required_cols} are missing in '{USERS_FILE}'.")
            return None

        # Clean search input and dataframe column to prevent spacing mismatches
        df["user_id_clean"] = df["user_id"].astype(str).str.strip()
        target_user = user_id.strip()

        # Find the row matching the user_id
        user_row = df[df["user_id_clean"] == target_user]

        if user_row.empty:
            print(f"[ERROR] Access Denied: User ID '{user_id}' was not found in our permissions database.")
            return None

        # Extract role and allowed projects
        role = str(user_row.iloc[0]["user_role"]).strip()
        allowed_projects_str = str(user_row.iloc[0]["allowed_projects"]).strip()

        # Parse projects string to a list of project IDs (handling comma separated values)
        allowed_projects = [proj.strip() for proj in allowed_projects_str.split(",") if proj.strip()]

        print(f"[SUCCESS] User '{user_id}' verified successfully.")
        return role, allowed_projects

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while verifying user session: {str(e)}")
        return None


if __name__ == "__main__":
    # Ensure database files exist
    setup_database()
    
    print("\n" + "="*60)
    print("        RUNNING AUTOMATIC USER SESSION VERIFICATION TESTS")
    print("="*60)
    
    # Test Case 1: user_elec_456
    print("\n[TEST 1] Verifying Electrical Subcontractor (user_elec_456):")
    res1 = verify_user_session("user_elec_456")
    if res1:
        role, projects = res1
        print(f"  - Role: {role}")
        print(f"  - Allowed Projects: {projects}")
    else:
        print("  - Result: Failed to verify user.")
        
    # Test Case 2: user_plumb_789
    print("\n[TEST 2] Verifying Plumbing Subcontractor (user_plumb_789):")
    res2 = verify_user_session("user_plumb_789")
    if res2:
        role, projects = res2
        print(f"  - Role: {role}")
        print(f"  - Allowed Projects: {projects}")
    else:
        print("  - Result: Failed to verify user.")
        
    # Test Case 3: Fake ID user_unknown_999
    print("\n[TEST 3] Verifying Unknown User (user_unknown_999):")
    res3 = verify_user_session("user_unknown_999")
    if res3:
        role, projects = res3
        print(f"  - Role: {role}")
        print(f"  - Allowed Projects: {projects}")
    else:
        print("  - Verification blocked correctly (Expected behavior).")
        
    print("\n" + "="*60)
