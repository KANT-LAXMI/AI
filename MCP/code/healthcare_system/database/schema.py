"""
Database Schema Definition
Defines all tables and their relationships for the healthcare system
"""

import aiosqlite
import asyncio
from pathlib import Path
from config.settings import DATABASE_PATH, DATABASE_DIR

# ===============================
# SCHEMA DEFINITIONS
# ===============================

# Table: Patients
PATIENTS_TABLE = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    date_of_birth DATE NOT NULL,
    gender TEXT CHECK(gender IN ('Male', 'Female', 'Other')),
    blood_type TEXT,
    phone TEXT,
    email TEXT UNIQUE,
    address TEXT,
    emergency_contact TEXT,
    emergency_phone TEXT,
    insurance_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Table: Medical Records
MEDICAL_RECORDS_TABLE = """
CREATE TABLE IF NOT EXISTS medical_records (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    visit_date DATE NOT NULL,
    diagnosis TEXT NOT NULL,
    symptoms TEXT,
    treatment TEXT,
    prescription TEXT,
    notes TEXT,
    doctor_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients (patient_id) ON DELETE CASCADE
);
"""

# Table: Appointments
APPOINTMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS appointments (
    appointment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    appointment_date DATETIME NOT NULL,
    appointment_type TEXT NOT NULL,
    doctor_name TEXT,
    status TEXT DEFAULT 'Scheduled' CHECK(status IN ('Scheduled', 'Completed', 'Cancelled', 'No-Show')),
    reason TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients (patient_id) ON DELETE CASCADE
);
"""

# Table: Medications
MEDICATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS medications (
    medication_id INTEGER PRIMARY KEY AUTOINCREMENT,
    medication_name TEXT NOT NULL UNIQUE,
    generic_name TEXT,
    category TEXT,
    description TEXT,
    side_effects TEXT,
    interactions TEXT,
    dosage_info TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Table: Prescriptions
PRESCRIPTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS prescriptions (
    prescription_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    medication_id INTEGER NOT NULL,
    record_id INTEGER,
    dosage TEXT NOT NULL,
    frequency TEXT NOT NULL,
    duration TEXT,
    prescribed_date DATE NOT NULL,
    status TEXT DEFAULT 'Active' CHECK(status IN ('Active', 'Completed', 'Cancelled')),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients (patient_id) ON DELETE CASCADE,
    FOREIGN KEY (medication_id) REFERENCES medications (medication_id),
    FOREIGN KEY (record_id) REFERENCES medical_records (record_id)
);
"""

# Table: Lab Results
LAB_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS lab_results (
    lab_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    record_id INTEGER,
    test_name TEXT NOT NULL,
    test_date DATE NOT NULL,
    result_value TEXT NOT NULL,
    normal_range TEXT,
    unit TEXT,
    status TEXT CHECK(status IN ('Normal', 'Abnormal', 'Critical')),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients (patient_id) ON DELETE CASCADE,
    FOREIGN KEY (record_id) REFERENCES medical_records (record_id)
);
"""

# Table: Vitals
VITALS_TABLE = """
CREATE TABLE IF NOT EXISTS vitals (
    vital_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    record_id INTEGER,
    measurement_date DATETIME NOT NULL,
    blood_pressure_systolic INTEGER,
    blood_pressure_diastolic INTEGER,
    heart_rate INTEGER,
    temperature REAL,
    respiratory_rate INTEGER,
    oxygen_saturation INTEGER,
    weight REAL,
    height REAL,
    bmi REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients (patient_id) ON DELETE CASCADE,
    FOREIGN KEY (record_id) REFERENCES medical_records (record_id)
);
"""

# ===============================
# INDEXES FOR PERFORMANCE
# ===============================
INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_patients_email ON patients(email);",
    "CREATE INDEX IF NOT EXISTS idx_patients_name ON patients(last_name, first_name);",
    "CREATE INDEX IF NOT EXISTS idx_medical_records_patient ON medical_records(patient_id);",
    "CREATE INDEX IF NOT EXISTS idx_appointments_patient ON appointments(patient_id);",
    "CREATE INDEX IF NOT EXISTS idx_appointments_date ON appointments(appointment_date);",
    "CREATE INDEX IF NOT EXISTS idx_prescriptions_patient ON prescriptions(patient_id);",
    "CREATE INDEX IF NOT EXISTS idx_lab_results_patient ON lab_results(patient_id);",
    "CREATE INDEX IF NOT EXISTS idx_vitals_patient ON vitals(patient_id);",
]

# ===============================
# SAMPLE DATA
# ===============================
SAMPLE_PATIENTS = """
INSERT OR IGNORE INTO patients (patient_id, first_name, last_name, date_of_birth, gender, blood_type, phone, email, address)
VALUES 
    (1, 'John', 'Doe', '1985-06-15', 'Male', 'O+', '555-0101', 'john.doe@email.com', '123 Main St'),
    (2, 'Jane', 'Smith', '1990-03-22', 'Female', 'A+', '555-0102', 'jane.smith@email.com', '456 Oak Ave'),
    (3, 'Robert', 'Johnson', '1978-11-08', 'Male', 'B+', '555-0103', 'robert.j@email.com', '789 Pine Rd'),
    (4, 'Emily', 'Davis', '1995-07-30', 'Female', 'AB-', '555-0104', 'emily.d@email.com', '321 Elm St');
"""

SAMPLE_MEDICATIONS = """
INSERT OR IGNORE INTO medications (medication_name, generic_name, category, description, side_effects)
VALUES 
    ('Aspirin', 'Acetylsalicylic Acid', 'Pain Reliever', 'Used to reduce pain and inflammation', 'Stomach upset, heartburn'),
    ('Lisinopril', 'Lisinopril', 'ACE Inhibitor', 'Used to treat high blood pressure', 'Dizziness, cough'),
    ('Metformin', 'Metformin HCL', 'Antidiabetic', 'Used to control blood sugar in diabetes', 'Nausea, diarrhea'),
    ('Amoxicillin', 'Amoxicillin', 'Antibiotic', 'Used to treat bacterial infections', 'Rash, diarrhea');
"""

SAMPLE_MEDICAL_RECORDS = """
INSERT OR IGNORE INTO medical_records (patient_id, visit_date, diagnosis, symptoms, treatment, doctor_name)
VALUES 
    (1, '2024-01-15', 'Hypertension', 'High blood pressure, headache', 'Prescribed Lisinopril', 'Dr. Sarah Wilson'),
    (2, '2024-02-20', 'Type 2 Diabetes', 'Increased thirst, frequent urination', 'Prescribed Metformin', 'Dr. Michael Chen'),
    (3, '2024-03-10', 'Acute Bronchitis', 'Cough, chest pain', 'Prescribed Amoxicillin', 'Dr. Sarah Wilson');
"""

# ===============================
# DATABASE INITIALIZATION
# ===============================
async def init_database():
    """
    Initialize the database with all tables, indexes, and sample data
    """
    # Ensure database directory exists
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    
    async with aiosqlite.connect(DATABASE_PATH) as db:
        # Enable foreign keys
        await db.execute("PRAGMA foreign_keys = ON;")
        
        # Create tables
        await db.execute(PATIENTS_TABLE)
        await db.execute(MEDICAL_RECORDS_TABLE)
        await db.execute(APPOINTMENTS_TABLE)
        await db.execute(MEDICATIONS_TABLE)
        await db.execute(PRESCRIPTIONS_TABLE)
        await db.execute(LAB_RESULTS_TABLE)
        await db.execute(VITALS_TABLE)
        
        # Create indexes
        for index in INDEXES:
            await db.execute(index)
        
        # Insert sample data
        await db.executescript(SAMPLE_PATIENTS)
        await db.executescript(SAMPLE_MEDICATIONS)
        await db.executescript(SAMPLE_MEDICAL_RECORDS)
        
        await db.commit()
        
        print(f"✅ Database initialized at: {DATABASE_PATH}")

# ===============================
# MAIN EXECUTION
# ===============================
if __name__ == "__main__":
    asyncio.run(init_database())