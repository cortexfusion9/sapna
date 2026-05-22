import psycopg2
import random

def initialize_database():
    try:
        # Connection setup
        conn = psycopg2.connect("dbname=gc_subsc user=postgres password=Sapna123 host=127.0.0.1")
        cur = conn.cursor()
        
        # 1. Reset tables with RESTART IDENTITY to reset IDs to 1
        cur.execute("TRUNCATE TABLE bid_management, pm_projects, crm_deals, user_subscriptions, users RESTART IDENTITY CASCADE;")
        
        # Helper data for realism
        project_types = ["Renovation", "New Construction", "Demolition", "Landscape", "Interior Design", "Structural Repair"]
        client_names = ["Apex Solutions", "BuildIt Corp", "Summit Realty", "Urban Dynamics", "GreenBuild Ltd"]
        
        print("🚀 Starting database seeding...")

        # 2. Loop through 30 users
        for i in range(1, 31):
            # Create user
            cur.execute("INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s) RETURNING user_id", 
                        (f"user_{i}", f"user{i}@test.com", "hash123"))
            uid = cur.fetchone()[0]
            
            # Subscription: 80% chance to have active MORGAN access
            has_access = random.random() < 0.8
            cur.execute("INSERT INTO user_subscriptions (user_id, module_name, is_active) VALUES (%s, %s, %s)", 
                        (uid, 'MORGAN', has_access))
            
            # Projects: Each user gets 1 to 3 random projects
            num_projects = random.randint(1, 3)
            for _ in range(num_projects):
                cur.execute("INSERT INTO pm_projects (user_id, project_name, budget, progress) VALUES (%s, %s, %s, %s)", 
                            (uid, f"{random.choice(project_types)} Phase {random.randint(1, 9)}", random.randint(50000, 750000), random.randint(5, 95)))
            
            # Bids: Each user gets 1 to 2 random bids
            num_bids = random.randint(1, 2)
            for _ in range(num_bids):
                cur.execute("INSERT INTO bid_management (user_id, proposal_name, amount, status) VALUES (%s, %s, %s, %s)", 
                            (uid, f"Proposal-{random.randint(100, 999)}", random.randint(100000, 1000000), random.choice(['Pending', 'Accepted', 'Rejected'])))

            # Deals: Each user gets 1 deal
            cur.execute("INSERT INTO crm_deals (user_id, client_name, deal_value, status) VALUES (%s, %s, %s, %s)", 
                        (uid, random.choice(client_names), random.randint(20000, 500000), random.choice(['Active', 'Closed', 'Negotiation'])))

        conn.commit()
        print(f"✅ Seeding successful! 30 users with varied projects and access levels created.")
        cur.close(); conn.close()
        
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    initialize_database()