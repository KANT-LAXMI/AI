"""
Medical MCP Server
Handles patient records, appointments, and medical history queries
Uses MCP protocol to expose database operations as tools
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP
from database.operations import DatabaseOperations
import json
from datetime import datetime
from typing import Optional

# ===============================
# MCP SERVER INITIALIZATION
# ===============================
mcp = FastMCP("medical")
db_ops = DatabaseOperations()

# ===============================
# PATIENT MANAGEMENT TOOLS
# ===============================

@mcp.tool()
async def search_patient(search_term: str) -> str:
    """
    Search for patients by name or email.
    
    Args:
        search_term: Name or email to search for (partial matches work)
    
    Returns:
        JSON string with list of matching patients
    """
    try:
        patients = await db_ops.search_patients(search_term)
        
        if not patients:
            return json.dumps({"status": "not_found", "message": f"No patients found matching '{search_term}'"})
        
        # Format patient data for readability
        formatted_patients = []
        for p in patients:
            formatted_patients.append({
                "patient_id": p['patient_id'],
                "name": f"{p['first_name']} {p['last_name']}",
                "dob": p['date_of_birth'],
                "gender": p['gender'],
                "blood_type": p['blood_type'],
                "email": p['email'],
                "phone": p['phone']
            })
        
        return json.dumps({
            "status": "success",
            "count": len(formatted_patients),
            "patients": formatted_patients
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
async def get_patient_details(patient_id: int) -> str:
    """
    Get complete details for a specific patient including summary statistics.
    
    Args:
        patient_id: The patient's ID number
    
    Returns:
        JSON string with patient details and summary
    """
    try:
        summary = await db_ops.get_patient_summary(patient_id)
        
        if not summary['patient']:
            return json.dumps({"status": "not_found", "message": f"Patient ID {patient_id} not found"})
        
        return json.dumps({
            "status": "success",
            "patient_info": summary['patient'],
            "statistics": {
                "total_medical_records": summary['total_records'],
                "total_appointments": summary['total_appointments'],
                "active_prescriptions": summary['active_prescriptions'],
                "last_visit": summary['last_visit']
            }
        }, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
async def get_medical_history(patient_id: int) -> str:
    """
    Get complete medical history for a patient including all past visits and diagnoses.
    
    Args:
        patient_id: The patient's ID number
    
    Returns:
        JSON string with medical history records
    """
    try:
        history = await db_ops.get_patient_medical_history(patient_id)
        
        if not history:
            return json.dumps({
                "status": "success",
                "message": "No medical records found for this patient",
                "records": []
            })
        
        return json.dumps({
            "status": "success",
            "count": len(history),
            "records": history
        }, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ===============================
# APPOINTMENT MANAGEMENT TOOLS
# ===============================

@mcp.tool()
async def get_appointments(patient_id: Optional[int] = None, date: Optional[str] = None) -> str:
    """
    Get appointments with optional filters.
    
    Args:
        patient_id: Filter by patient ID (optional)
        date: Filter by date (YYYY-MM-DD format, returns appointments from this date onwards) (optional)
    
    Returns:
        JSON string with appointment list
    """
    try:
        appointments = await db_ops.get_appointments(patient_id, date)
        
        if not appointments:
            return json.dumps({
                "status": "success",
                "message": "No appointments found",
                "appointments": []
            })
        
        # Format appointments for readability
        formatted = []
        for apt in appointments:
            formatted.append({
                "appointment_id": apt['appointment_id'],
                "patient": f"{apt['first_name']} {apt['last_name']}",
                "patient_id": apt['patient_id'],
                "date": apt['appointment_date'],
                "type": apt['appointment_type'],
                "doctor": apt['doctor_name'],
                "status": apt['status'],
                "reason": apt['reason']
            })
        
        return json.dumps({
            "status": "success",
            "count": len(formatted),
            "appointments": formatted
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
async def schedule_appointment(patient_id: int, appointment_date: str, 
                               appointment_type: str, doctor_name: str, 
                               reason: str) -> str:
    """
    Schedule a new appointment for a patient.
    
    Args:
        patient_id: The patient's ID number
        appointment_date: Date and time (YYYY-MM-DD HH:MM format)
        appointment_type: Type of appointment (e.g., "General Checkup", "Follow-up", "Emergency")
        doctor_name: Name of the doctor
        reason: Reason for the appointment
    
    Returns:
        JSON string with confirmation and appointment ID
    """
    try:
        appointment_id = await db_ops.schedule_appointment({
            'patient_id': patient_id,
            'appointment_date': appointment_date,
            'appointment_type': appointment_type,
            'doctor_name': doctor_name,
            'reason': reason,
            'status': 'Scheduled'
        })
        
        return json.dumps({
            "status": "success",
            "message": "Appointment scheduled successfully",
            "appointment_id": appointment_id,
            "details": {
                "patient_id": patient_id,
                "date": appointment_date,
                "type": appointment_type,
                "doctor": doctor_name
            }
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ===============================
# PRESCRIPTION MANAGEMENT TOOLS
# ===============================

@mcp.tool()
async def get_prescriptions(patient_id: int, active_only: bool = False) -> str:
    """
    Get all prescriptions for a patient.
    
    Args:
        patient_id: The patient's ID number
        active_only: If True, only return active prescriptions (default: False)
    
    Returns:
        JSON string with prescription list including medication details
    """
    try:
        prescriptions = await db_ops.get_patient_prescriptions(patient_id, active_only)
        
        if not prescriptions:
            status_text = "active" if active_only else ""
            return json.dumps({
                "status": "success",
                "message": f"No {status_text} prescriptions found".strip(),
                "prescriptions": []
            })
        
        # Format prescriptions
        formatted = []
        for rx in prescriptions:
            formatted.append({
                "prescription_id": rx['prescription_id'],
                "medication": rx['medication_name'],
                "generic_name": rx['generic_name'],
                "dosage": rx['dosage'],
                "frequency": rx['frequency'],
                "duration": rx['duration'],
                "prescribed_date": rx['prescribed_date'],
                "status": rx['status'],
                "side_effects": rx['side_effects']
            })
        
        return json.dumps({
            "status": "success",
            "count": len(formatted),
            "prescriptions": formatted
        }, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ===============================
# VITALS TRACKING TOOLS
# ===============================

@mcp.tool()
async def get_vitals_history(patient_id: int, limit: int = 10) -> str:
    """
    Get vital signs history for a patient.
    
    Args:
        patient_id: The patient's ID number
        limit: Maximum number of records to return (default: 10)
    
    Returns:
        JSON string with vitals history
    """
    try:
        vitals = await db_ops.get_patient_vitals(patient_id, limit)
        
        if not vitals:
            return json.dumps({
                "status": "success",
                "message": "No vital signs recorded for this patient",
                "vitals": []
            })
        
        return json.dumps({
            "status": "success",
            "count": len(vitals),
            "vitals": vitals
        }, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
async def record_vitals(patient_id: int, measurement_date: str,
                       blood_pressure_systolic: Optional[int] = None,
                       blood_pressure_diastolic: Optional[int] = None,
                       heart_rate: Optional[int] = None,
                       temperature: Optional[float] = None,
                       oxygen_saturation: Optional[int] = None,
                       weight: Optional[float] = None) -> str:
    """
    Record new vital signs for a patient.
    
    Args:
        patient_id: The patient's ID number
        measurement_date: Date and time of measurement (YYYY-MM-DD HH:MM format)
        blood_pressure_systolic: Systolic BP (optional)
        blood_pressure_diastolic: Diastolic BP (optional)
        heart_rate: Heart rate in BPM (optional)
        temperature: Body temperature in Fahrenheit (optional)
        oxygen_saturation: SpO2 percentage (optional)
        weight: Weight in pounds (optional)
    
    Returns:
        JSON string with confirmation
    """
    try:
        vital_id = await db_ops.add_vitals({
            'patient_id': patient_id,
            'measurement_date': measurement_date,
            'blood_pressure_systolic': blood_pressure_systolic,
            'blood_pressure_diastolic': blood_pressure_diastolic,
            'heart_rate': heart_rate,
            'temperature': temperature,
            'oxygen_saturation': oxygen_saturation,
            'weight': weight,
            'record_id': None
        })
        
        return json.dumps({
            "status": "success",
            "message": "Vital signs recorded successfully",
            "vital_id": vital_id
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ===============================
# ANALYTICS TOOLS
# ===============================

@mcp.tool()
async def get_system_statistics() -> str:
    """
    Get overall system statistics including patient count, appointments, etc.
    
    Returns:
        JSON string with system-wide statistics
    """
    try:
        stats = await db_ops.get_statistics()
        
        return json.dumps({
            "status": "success",
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ===============================
# TRANSPORT LAYER
# ===============================
if __name__ == "__main__":
    # Run the MCP server with stdio transport
    mcp.run(transport="stdio")