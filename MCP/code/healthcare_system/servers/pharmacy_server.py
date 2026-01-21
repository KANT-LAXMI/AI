"""
Pharmacy MCP Server
Handles medication information, drug interactions, and prescription management
Includes web search capabilities for drug information
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP
from database.operations import DatabaseOperations
import httpx
import json
from typing import Optional, List
from bs4 import BeautifulSoup

# ===============================
# MCP SERVER INITIALIZATION
# ===============================
mcp = FastMCP("pharmacy")
db_ops = DatabaseOperations()

USER_AGENT = "healthcare-system/1.0"

# ===============================
# MEDICATION DATABASE TOOLS
# ===============================

@mcp.tool()
async def search_medications(search_term: str) -> str:
    """
    Search for medications in the database by name.
    
    Args:
        search_term: Medication name or generic name to search for
    
    Returns:
        JSON string with matching medications
    """
    try:
        medications = await db_ops.get_all_medications()
        
        # Filter medications by search term
        search_lower = search_term.lower()
        matches = [
            med for med in medications
            if search_lower in med['medication_name'].lower() or
               (med['generic_name'] and search_lower in med['generic_name'].lower())
        ]
        
        if not matches:
            return json.dumps({
                "status": "not_found",
                "message": f"No medications found matching '{search_term}'"
            })
        
        return json.dumps({
            "status": "success",
            "count": len(matches),
            "medications": matches
        }, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
async def get_medication_info(medication_id: int) -> str:
    """
    Get detailed information about a specific medication.
    
    Args:
        medication_id: The medication's ID number
    
    Returns:
        JSON string with complete medication information
    """
    try:
        medications = await db_ops.get_all_medications()
        medication = next((m for m in medications if m['medication_id'] == medication_id), None)
        
        if not medication:
            return json.dumps({
                "status": "not_found",
                "message": f"Medication ID {medication_id} not found"
            })
        
        return json.dumps({
            "status": "success",
            "medication": medication
        }, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ===============================
# DRUG INTERACTION CHECKER
# ===============================

@mcp.tool()
async def check_drug_interactions(patient_id: int, new_medication_name: str) -> str:
    """
    Check for potential drug interactions between a patient's current medications and a new medication.
    This is a simplified check - in production, this would use a comprehensive drug interaction database.
    
    Args:
        patient_id: The patient's ID number
        new_medication_name: Name of the medication being considered
    
    Returns:
        JSON string with interaction warnings and current medications
    """
    try:
        # Get patient's current active prescriptions
        prescriptions = await db_ops.get_patient_prescriptions(patient_id, active_only=True)
        
        if not prescriptions:
            return json.dumps({
                "status": "success",
                "message": "Patient has no active prescriptions",
                "new_medication": new_medication_name,
                "interactions": [],
                "recommendation": "No interactions detected (patient not on other medications)"
            })
        
        # Extract current medication names
        current_meds = [rx['medication_name'] for rx in prescriptions]
        
        # Known interaction pairs (simplified for demonstration)
        # In production, use a comprehensive drug interaction database
        known_interactions = {
            "Aspirin": ["Warfarin", "Ibuprofen"],
            "Warfarin": ["Aspirin", "Vitamin K"],
            "Metformin": ["Alcohol"],
            "Lisinopril": ["Potassium supplements"],
            "Amoxicillin": ["Methotrexate"]
        }
        
        # Check for interactions
        interactions = []
        new_med_upper = new_medication_name.upper()
        
        for current_med in current_meds:
            current_med_upper = current_med.upper()
            
            # Check if new medication interacts with current medication
            if current_med_upper in known_interactions.get(new_medication_name, []):
                interactions.append({
                    "medication_pair": f"{new_medication_name} + {current_med}",
                    "severity": "Moderate",
                    "description": f"Potential interaction between {new_medication_name} and {current_med}"
                })
            
            # Check reverse
            if new_med_upper in known_interactions.get(current_med, []):
                interactions.append({
                    "medication_pair": f"{current_med} + {new_medication_name}",
                    "severity": "Moderate",
                    "description": f"Potential interaction between {current_med} and {new_medication_name}"
                })
        
        recommendation = (
            "⚠️ CAUTION: Potential drug interactions detected. Consult pharmacist or physician."
            if interactions else
            "✅ No known interactions detected with current medications."
        )
        
        return json.dumps({
            "status": "success",
            "new_medication": new_medication_name,
            "current_medications": current_meds,
            "interactions_found": len(interactions),
            "interactions": interactions,
            "recommendation": recommendation
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ===============================
# PRESCRIPTION MANAGEMENT
# ===============================

@mcp.tool()
async def add_prescription(patient_id: int, medication_name: str, dosage: str,
                          frequency: str, duration: str, prescribed_date: str) -> str:
    """
    Add a new prescription for a patient.
    
    Args:
        patient_id: The patient's ID number
        medication_name: Name of the medication
        dosage: Dosage (e.g., "500mg", "10ml")
        frequency: Frequency (e.g., "Twice daily", "Every 8 hours")
        duration: Duration (e.g., "7 days", "1 month")
        prescribed_date: Date prescribed (YYYY-MM-DD format)
    
    Returns:
        JSON string with confirmation and prescription ID
    """
    try:
        # Find medication ID by name
        medications = await db_ops.get_all_medications()
        medication = next(
            (m for m in medications if m['medication_name'].lower() == medication_name.lower()),
            None
        )
        
        if not medication:
            return json.dumps({
                "status": "error",
                "message": f"Medication '{medication_name}' not found in database"
            })
        
        # Add prescription
        prescription_id = await db_ops.add_prescription({
            'patient_id': patient_id,
            'medication_id': medication['medication_id'],
            'dosage': dosage,
            'frequency': frequency,
            'duration': duration,
            'prescribed_date': prescribed_date,
            'status': 'Active',
            'record_id': None
        })
        
        return json.dumps({
            "status": "success",
            "message": "Prescription added successfully",
            "prescription_id": prescription_id,
            "details": {
                "patient_id": patient_id,
                "medication": medication_name,
                "dosage": dosage,
                "frequency": frequency,
                "duration": duration
            }
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ===============================
# WEB SEARCH FOR DRUG INFORMATION
# ===============================

async def search_drug_info_web(drug_name: str) -> dict:
    """
    Search the web for drug information using a search engine.
    This is a simplified version - in production, use medical databases like DrugBank or FDA.
    """
    search_query = f"{drug_name} medication information uses side effects"
    
    # In production, use SERPER_API_KEY or similar
    # For demonstration, return structured response
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
        try:
            # This is a placeholder - in production, integrate with medical APIs
            # Example: FDA Drug Database, DrugBank, or RxNorm
            return {
                "drug_name": drug_name,
                "source": "web_search",
                "info": f"Search results for {drug_name} (integrate real API in production)"
            }
        except Exception as e:
            return {"error": str(e)}


@mcp.tool()
async def get_drug_info_online(drug_name: str) -> str:
    """
    Search online medical databases for comprehensive drug information.
    Note: This is a demonstration tool. In production, integrate with FDA, DrugBank, or RxNorm APIs.
    
    Args:
        drug_name: Name of the drug to search for
    
    Returns:
        JSON string with drug information from online sources
    """
    try:
        # First check local database
        medications = await db_ops.get_all_medications()
        local_match = next(
            (m for m in medications if m['medication_name'].lower() == drug_name.lower()),
            None
        )
        
        response = {
            "status": "success",
            "drug_name": drug_name,
            "local_database": local_match if local_match else "Not found in local database",
            "note": "Online search capability - integrate with FDA/DrugBank API in production",
            "recommendation": "For comprehensive drug information, consult FDA database or DrugBank"
        }
        
        return json.dumps(response, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ===============================
# MEDICATION ADHERENCE TOOLS
# ===============================

@mcp.tool()
async def check_medication_adherence(patient_id: int) -> str:
    """
    Check patient's medication adherence by analyzing prescription history.
    
    Args:
        patient_id: The patient's ID number
    
    Returns:
        JSON string with adherence information and recommendations
    """
    try:
        prescriptions = await db_ops.get_patient_prescriptions(patient_id)
        
        if not prescriptions:
            return json.dumps({
                "status": "success",
                "message": "No prescription history for this patient",
                "adherence_score": None
            })
        
        # Calculate simple adherence metrics
        total = len(prescriptions)
        active = sum(1 for rx in prescriptions if rx['status'] == 'Active')
        completed = sum(1 for rx in prescriptions if rx['status'] == 'Completed')
        
        adherence_score = (completed / total * 100) if total > 0 else 0
        
        # Determine adherence level
        if adherence_score >= 80:
            adherence_level = "Excellent"
            recommendation = "Patient shows good medication adherence"
        elif adherence_score >= 60:
            adherence_level = "Good"
            recommendation = "Patient shows moderate adherence - consider follow-up"
        else:
            adherence_level = "Needs Improvement"
            recommendation = "⚠️ Low adherence detected - schedule counseling session"
        
        return json.dumps({
            "status": "success",
            "patient_id": patient_id,
            "total_prescriptions": total,
            "active_prescriptions": active,
            "completed_prescriptions": completed,
            "adherence_score": round(adherence_score, 1),
            "adherence_level": adherence_level,
            "recommendation": recommendation
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ===============================
# TRANSPORT LAYER
# ===============================
if __name__ == "__main__":
    # Run the MCP server with stdio transport
    mcp.run(transport="stdio")