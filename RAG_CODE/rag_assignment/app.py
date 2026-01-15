from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, EmailStr, constr
import sqlite3
import uuid
from datetime import datetime

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Complaint Creation API")

# -------------------------------
# Database connection helper
# -------------------------------
DB_FILE = "rag_chatbot_system.db"

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn

# -------------------------------
# Request model
# -------------------------------
class ComplaintCreate(BaseModel):
    name: constr(strip_whitespace=True, min_length=1)
    phone_number: constr(strip_whitespace=True, min_length=7, max_length=15)
    email: EmailStr
    complaint_details: constr(strip_whitespace=True, min_length=5)

# -------------------------------
# POST /complaints
# -------------------------------
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
@app.post("/complaints")
def create_complaint(complaint: ComplaintCreate):
    print("=== POST /complaints called ===")
    print(f"Received data: {complaint.dict()}")

    complaint_id = str(uuid.uuid4())  # Unique complaint ID
    created_at = datetime.now().isoformat()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO complaints (complaint_id, name, phone_number, email, complaint_details, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            complaint_id,
            complaint.name,
            complaint.phone_number,
            complaint.email,
            complaint.complaint_details,
            created_at
        ))
        conn.commit()
        print(f"Complaint inserted into DB with ID: {complaint_id}")
    except Exception as e:
        conn.close()
        print(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    conn.close()
    print("Returning response for POST /complaints")
    return {
        "complaint_id": complaint_id,
        "message": "Complaint created successfully"
    }

# -------------------------------
# GET /complaints/{complaint_id}
# -------------------------------
@app.get("/complaints/{complaint_id}")
def get_complaint(complaint_id: str):
    print(f"=== GET /complaints/{complaint_id} called ===")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM complaints WHERE complaint_id = ?", (complaint_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        print(f"Complaint ID {complaint_id} not found in DB")
        raise HTTPException(status_code=404, detail="Complaint not found")
    
    print(f"Complaint found: {dict(row)}")
    return {
        "complaint_id": row["complaint_id"],
        "name": row["name"],
        "phone_number": row["phone_number"],
        "email": row["email"],
        "complaint_details": row["complaint_details"],
        "created_at": row["created_at"]
    }

# -------------------------------
# Run FastAPI with: uvicorn api:app --reload
# -------------------------------
