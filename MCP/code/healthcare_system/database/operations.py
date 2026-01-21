"""
Database Operations
CRUD operations and complex queries for the healthcare system
"""

import aiosqlite
from typing import List, Dict, Optional, Any
from datetime import datetime, date
from config.settings import DATABASE_PATH


class DatabaseOperations:
    """
    Handles all database operations for the healthcare system
    """
    
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
    
    # ===============================
    # PATIENT OPERATIONS
    # ===============================
    
    async def add_patient(self, patient_data: Dict[str, Any]) -> int:
        """Add a new patient to the database"""
        query = """
        INSERT INTO patients (first_name, last_name, date_of_birth, gender, blood_type, 
                             phone, email, address, emergency_contact, emergency_phone, insurance_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, (
                patient_data.get('first_name'),
                patient_data.get('last_name'),
                patient_data.get('date_of_birth'),
                patient_data.get('gender'),
                patient_data.get('blood_type'),
                patient_data.get('phone'),
                patient_data.get('email'),
                patient_data.get('address'),
                patient_data.get('emergency_contact'),
                patient_data.get('emergency_phone'),
                patient_data.get('insurance_id')
            ))
            await db.commit()
            return cursor.lastrowid
    
    async def get_patient(self, patient_id: int) -> Optional[Dict[str, Any]]:
        """Get patient details by ID"""
        query = "SELECT * FROM patients WHERE patient_id = ?"
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, (patient_id,)) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None
    
    async def search_patients(self, search_term: str) -> List[Dict[str, Any]]:
        """Search patients by name or email"""
        query = """
        SELECT * FROM patients 
        WHERE first_name LIKE ? OR last_name LIKE ? OR email LIKE ?
        ORDER BY last_name, first_name
        """
        search_pattern = f"%{search_term}%"
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, (search_pattern, search_pattern, search_pattern)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def get_all_patients(self) -> List[Dict[str, Any]]:
        """Get all patients"""
        query = "SELECT * FROM patients ORDER BY last_name, first_name"
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def update_patient(self, patient_id: int, patient_data: Dict[str, Any]) -> bool:
        """Update patient information"""
        query = """
        UPDATE patients 
        SET first_name=?, last_name=?, date_of_birth=?, gender=?, blood_type=?,
            phone=?, email=?, address=?, emergency_contact=?, emergency_phone=?, 
            insurance_id=?, updated_at=CURRENT_TIMESTAMP
        WHERE patient_id=?
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(query, (
                patient_data.get('first_name'),
                patient_data.get('last_name'),
                patient_data.get('date_of_birth'),
                patient_data.get('gender'),
                patient_data.get('blood_type'),
                patient_data.get('phone'),
                patient_data.get('email'),
                patient_data.get('address'),
                patient_data.get('emergency_contact'),
                patient_data.get('emergency_phone'),
                patient_data.get('insurance_id'),
                patient_id
            ))
            await db.commit()
            return True
    
    # ===============================
    # MEDICAL RECORD OPERATIONS
    # ===============================
    
    async def add_medical_record(self, record_data: Dict[str, Any]) -> int:
        """Add a new medical record"""
        query = """
        INSERT INTO medical_records (patient_id, visit_date, diagnosis, symptoms, 
                                     treatment, prescription, notes, doctor_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, (
                record_data.get('patient_id'),
                record_data.get('visit_date'),
                record_data.get('diagnosis'),
                record_data.get('symptoms'),
                record_data.get('treatment'),
                record_data.get('prescription'),
                record_data.get('notes'),
                record_data.get('doctor_name')
            ))
            await db.commit()
            return cursor.lastrowid
    
    async def get_patient_medical_history(self, patient_id: int) -> List[Dict[str, Any]]:
        """Get complete medical history for a patient"""
        query = """
        SELECT * FROM medical_records 
        WHERE patient_id = ? 
        ORDER BY visit_date DESC
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, (patient_id,)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    # ===============================
    # APPOINTMENT OPERATIONS
    # ===============================
    
    async def schedule_appointment(self, appointment_data: Dict[str, Any]) -> int:
        """Schedule a new appointment"""
        query = """
        INSERT INTO appointments (patient_id, appointment_date, appointment_type, 
                                 doctor_name, status, reason, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, (
                appointment_data.get('patient_id'),
                appointment_data.get('appointment_date'),
                appointment_data.get('appointment_type'),
                appointment_data.get('doctor_name'),
                appointment_data.get('status', 'Scheduled'),
                appointment_data.get('reason'),
                appointment_data.get('notes')
            ))
            await db.commit()
            return cursor.lastrowid
    
    async def get_appointments(self, patient_id: Optional[int] = None, 
                              start_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get appointments with optional filters"""
        if patient_id and start_date:
            query = """
            SELECT a.*, p.first_name, p.last_name 
            FROM appointments a
            JOIN patients p ON a.patient_id = p.patient_id
            WHERE a.patient_id = ? AND DATE(a.appointment_date) >= DATE(?)
            ORDER BY a.appointment_date
            """
            params = (patient_id, start_date)
        elif patient_id:
            query = """
            SELECT a.*, p.first_name, p.last_name 
            FROM appointments a
            JOIN patients p ON a.patient_id = p.patient_id
            WHERE a.patient_id = ?
            ORDER BY a.appointment_date DESC
            """
            params = (patient_id,)
        elif start_date:
            query = """
            SELECT a.*, p.first_name, p.last_name 
            FROM appointments a
            JOIN patients p ON a.patient_id = p.patient_id
            WHERE DATE(a.appointment_date) >= DATE(?)
            ORDER BY a.appointment_date
            """
            params = (start_date,)
        else:
            query = """
            SELECT a.*, p.first_name, p.last_name 
            FROM appointments a
            JOIN patients p ON a.patient_id = p.patient_id
            ORDER BY a.appointment_date DESC
            LIMIT 100
            """
            params = ()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def update_appointment_status(self, appointment_id: int, status: str) -> bool:
        """Update appointment status"""
        query = "UPDATE appointments SET status = ? WHERE appointment_id = ?"
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(query, (status, appointment_id))
            await db.commit()
            return True
    
    # ===============================
    # MEDICATION & PRESCRIPTION OPERATIONS
    # ===============================
    
    async def get_all_medications(self) -> List[Dict[str, Any]]:
        """Get all medications"""
        query = "SELECT * FROM medications ORDER BY medication_name"
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def add_prescription(self, prescription_data: Dict[str, Any]) -> int:
        """Add a new prescription"""
        query = """
        INSERT INTO prescriptions (patient_id, medication_id, record_id, dosage, 
                                  frequency, duration, prescribed_date, status, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, (
                prescription_data.get('patient_id'),
                prescription_data.get('medication_id'),
                prescription_data.get('record_id'),
                prescription_data.get('dosage'),
                prescription_data.get('frequency'),
                prescription_data.get('duration'),
                prescription_data.get('prescribed_date'),
                prescription_data.get('status', 'Active'),
                prescription_data.get('notes')
            ))
            await db.commit()
            return cursor.lastrowid
    
    async def get_patient_prescriptions(self, patient_id: int, 
                                       active_only: bool = False) -> List[Dict[str, Any]]:
        """Get patient prescriptions"""
        if active_only:
            query = """
            SELECT p.*, m.medication_name, m.generic_name, m.side_effects
            FROM prescriptions p
            JOIN medications m ON p.medication_id = m.medication_id
            WHERE p.patient_id = ? AND p.status = 'Active'
            ORDER BY p.prescribed_date DESC
            """
        else:
            query = """
            SELECT p.*, m.medication_name, m.generic_name, m.side_effects
            FROM prescriptions p
            JOIN medications m ON p.medication_id = m.medication_id
            WHERE p.patient_id = ?
            ORDER BY p.prescribed_date DESC
            """
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, (patient_id,)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    # ===============================
    # VITALS OPERATIONS
    # ===============================
    
    async def add_vitals(self, vitals_data: Dict[str, Any]) -> int:
        """Add vital signs measurement"""
        query = """
        INSERT INTO vitals (patient_id, record_id, measurement_date, 
                           blood_pressure_systolic, blood_pressure_diastolic,
                           heart_rate, temperature, respiratory_rate, 
                           oxygen_saturation, weight, height, bmi)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, (
                vitals_data.get('patient_id'),
                vitals_data.get('record_id'),
                vitals_data.get('measurement_date'),
                vitals_data.get('blood_pressure_systolic'),
                vitals_data.get('blood_pressure_diastolic'),
                vitals_data.get('heart_rate'),
                vitals_data.get('temperature'),
                vitals_data.get('respiratory_rate'),
                vitals_data.get('oxygen_saturation'),
                vitals_data.get('weight'),
                vitals_data.get('height'),
                vitals_data.get('bmi')
            ))
            await db.commit()
            return cursor.lastrowid
    
    async def get_patient_vitals(self, patient_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get patient vital signs history"""
        query = """
        SELECT * FROM vitals 
        WHERE patient_id = ? 
        ORDER BY measurement_date DESC 
        LIMIT ?
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, (patient_id, limit)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    # ===============================
    # ANALYTICS & REPORTING
    # ===============================
    
    async def get_patient_summary(self, patient_id: int) -> Dict[str, Any]:
        """Get comprehensive patient summary"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Get patient info
            patient = await self.get_patient(patient_id)
            
            # Get record counts
            async with db.execute(
                "SELECT COUNT(*) as count FROM medical_records WHERE patient_id = ?",
                (patient_id,)
            ) as cursor:
                record_count = (await cursor.fetchone())['count']
            
            # Get appointment counts
            async with db.execute(
                "SELECT COUNT(*) as count FROM appointments WHERE patient_id = ?",
                (patient_id,)
            ) as cursor:
                appointment_count = (await cursor.fetchone())['count']
            
            # Get active prescriptions count
            async with db.execute(
                "SELECT COUNT(*) as count FROM prescriptions WHERE patient_id = ? AND status = 'Active'",
                (patient_id,)
            ) as cursor:
                active_prescriptions = (await cursor.fetchone())['count']
            
            # Get last visit
            async with db.execute(
                "SELECT visit_date FROM medical_records WHERE patient_id = ? ORDER BY visit_date DESC LIMIT 1",
                (patient_id,)
            ) as cursor:
                last_visit_row = await cursor.fetchone()
                last_visit = last_visit_row['visit_date'] if last_visit_row else None
            
            return {
                'patient': patient,
                'total_records': record_count,
                'total_appointments': appointment_count,
                'active_prescriptions': active_prescriptions,
                'last_visit': last_visit
            }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            stats = {}
            
            # Total patients
            async with db.execute("SELECT COUNT(*) as count FROM patients") as cursor:
                stats['total_patients'] = (await cursor.fetchone())['count']
            
            # Total appointments
            async with db.execute("SELECT COUNT(*) as count FROM appointments") as cursor:
                stats['total_appointments'] = (await cursor.fetchone())['count']
            
            # Today's appointments
            async with db.execute(
                "SELECT COUNT(*) as count FROM appointments WHERE DATE(appointment_date) = DATE('now')"
            ) as cursor:
                stats['todays_appointments'] = (await cursor.fetchone())['count']
            
            # Active prescriptions
            async with db.execute(
                "SELECT COUNT(*) as count FROM prescriptions WHERE status = 'Active'"
            ) as cursor:
                stats['active_prescriptions'] = (await cursor.fetchone())['count']
            
            return stats