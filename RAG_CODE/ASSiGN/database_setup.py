import sqlite3
from datetime import datetime
import uuid

db_file = "rag_chatbot_system.db"

def setup_database():
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

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

    # Insert sample knowledge base data
    knowledge_entries = [
        ("Complaint Handling Policy",
         "Our company handles all customer complaints promptly. Complaints are acknowledged within 24 hours and resolved within 5 business days. You can track your complaint status using the unique complaint ID provided to you.",
         "policy_document.pdf"),
        
        ("Refund Policy",
         "Refunds can be requested within 14 days of purchase if the product is defective or not as described. To initiate a refund, please file a complaint with your order details. Refunds are processed within 7-10 business days after approval.",
         "refund_policy.txt"),
        
        ("Contact Support",
         "Customers can contact support via email at support@company.com or call our helpline at +1-800-555-0199. Our support team is available Monday to Friday, 9 AM to 6 PM EST.",
         "faq.txt"),
        
        ("Warranty Policy",
         "All products come with a 1-year warranty covering manufacturing defects. Warranty claims must be submitted with proof of purchase. Extended warranty options are available at the time of purchase.",
         "warranty.pdf"),
        
        ("Shipping Policy",
         "Orders are processed within 2 business days and shipping takes 3-7 business days depending on location. Express shipping is available for an additional fee. Track your order using the tracking number provided in your confirmation email.",
         "shipping_policy.txt"),
        
        ("Return Policy",
         "Items can be returned within 30 days of delivery for a full refund or exchange. Items must be unused and in original packaging. Return shipping costs may apply unless the item is defective.",
         "return_policy.txt"),
        
        ("Product Defects",
         "If you receive a defective product, please file a complaint immediately with photos of the defect. We will arrange for a replacement or full refund. Defective items do not require return shipping charges.",
         "quality_assurance.pdf"),
        
        ("Delayed Delivery",
         "If your delivery is delayed beyond the estimated delivery date, please check your tracking information first. If there are no updates for 3 days, file a complaint and we will investigate with the shipping carrier immediately.",
         "shipping_guidelines.txt")
    ]

    cursor.executemany("""
    INSERT OR IGNORE INTO knowledge_base (title, content, source)
    VALUES (?, ?, ?)
    """, knowledge_entries)

    # Insert sample complaints
    complaint_entries = [
        (str(uuid.uuid4()), "John Doe", "+1234567890", "john@example.com", "Product not working as expected. The device turns off randomly."),
        (str(uuid.uuid4()), "Jane Smith", "+1987654321", "jane@example.com", "Received damaged item. The box was crushed and product is broken."),
    ]

    cursor.executemany("""
    INSERT OR IGNORE INTO complaints (complaint_id, name, phone_number, email, complaint_details)
    VALUES (?, ?, ?, ?, ?)
    """, complaint_entries)

    conn.commit()
    conn.close()
    print(f"✅ Database '{db_file}' setup complete!")

if __name__ == "__main__":
    setup_database()
