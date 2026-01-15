from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr, constr
import sqlite3
import uuid
from datetime import datetime
import re

app = FastAPI(title="Complaint Creation API")

DB_FILE = "rag_chatbot_system.db"

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def validate_phone(phone: str) -> bool:
    """Validate phone number format"""
    pattern = r'^[+]?[\d\s\-()]{7,15}$'
    return bool(re.match(pattern, phone))

class ComplaintCreate(BaseModel):
    name: constr(strip_whitespace=True, min_length=1)
    phone_number: constr(strip_whitespace=True, min_length=7, max_length=20)
    email: EmailStr
    complaint_details: constr(strip_whitespace=True, min_length=5)

@app.get("/")
def read_root():
    return {"message": "Complaint API is running", "status": "OK"}

@app.post("/complaints")
def create_complaint(complaint: ComplaintCreate):
    if not validate_phone(complaint.phone_number):
        raise HTTPException(status_code=400, detail="Invalid phone number format")
    
    complaint_id = str(uuid.uuid4())
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
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    conn.close()
    return {
        "complaint_id": complaint_id,
        "message": "Complaint created successfully"
    }

@app.get("/complaints/{complaint_id}")
def get_complaint(complaint_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM complaints WHERE complaint_id = ?", (complaint_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        raise HTTPException(status_code=404, detail="Complaint not found")
    
    return {
        "complaint_id": row["complaint_id"],
        "name": row["name"],
        "phone_number": row["phone_number"],
        "email": row["email"],
        "complaint_details": row["complaint_details"],
        "created_at": row["created_at"]
    }