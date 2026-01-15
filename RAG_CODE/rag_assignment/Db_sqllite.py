import sqlite3
from datetime import datetime
import uuid

# -------------------------------
# Create/connect to SQLite database
# -------------------------------
db_file = "rag_chatbot_system.db"
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# -------------------------------
# Create tables
# -------------------------------

# Knowledge Base Table
cursor.execute("""
CREATE TABLE IF NOT EXISTS knowledge_base (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT,
    embedding BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# Complaints Table
cursor.execute("""
CREATE TABLE IF NOT EXISTS complaints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    phone_number TEXT NOT NULL,
    email TEXT NOT NULL,
    complaint_details TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# Chat Context Table
cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_context (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_message TEXT,
    bot_response TEXT,
    context_state TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# -------------------------------
# Insert sample data into knowledge_base
# -------------------------------
knowledge_entries = [
    ("Complaint Handling Policy",
     "Our company handles all customer complaints promptly. Complaints are acknowledged within 24 hours and resolved within 5 business days.",
     "policy_document.pdf"),
    
    ("Refund Policy",
     "Refunds can be requested within 14 days of purchase if the product is defective or not as described.",
     "refund_policy.txt"),
    
    ("Contact Support",
     "Customers can contact support via email at support@company.com or call our helpline at +1-800-555-0199.",
     "faq.txt"),
    
    ("Warranty Policy",
     "All products come with a 1-year warranty covering manufacturing defects. Warranty claims must be submitted with proof of purchase.",
     "warranty.pdf"),
    
    ("Shipping Policy",
     "Orders are processed within 2 business days and shipping takes 3-7 business days depending on location.",
     "shipping_policy.txt")
]

cursor.executemany("""
INSERT INTO knowledge_base (title, content, source)
VALUES (?, ?, ?)
""", knowledge_entries)

# -------------------------------
# Insert sample data into complaints
# -------------------------------
complaint_entries = [
    (str(uuid.uuid4()), "John Doe", "+1234567890", "john@example.com", "Product not working as expected."),
    (str(uuid.uuid4()), "Jane Smith", "+1987654321", "jane@example.com", "Received damaged item."),
    (str(uuid.uuid4()), "Alice Johnson", "+1122334455", "alice@example.com", "Refund not processed yet."),
    (str(uuid.uuid4()), "Bob Brown", "+1223344556", "bob@example.com", "Wrong product delivered."),
]

cursor.executemany("""
INSERT INTO complaints (complaint_id, name, phone_number, email, complaint_details)
VALUES (?, ?, ?, ?, ?)
""", complaint_entries)

# -------------------------------
# Commit changes and close connection
# -------------------------------
conn.commit()
conn.close()

print(f"✅ Database '{db_file}' created with tables and sample data!")
