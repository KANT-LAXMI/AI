"""
Helper Utilities
Common utility functions used across the application
"""

from datetime import datetime, date
from typing import Any, Dict, List
import json


# ===============================
# DATE/TIME UTILITIES
# ===============================
def format_date(date_obj: Any, format_str: str = "%Y-%m-%d") -> str:
    """
    Format a date object to string.
    
    Args:
        date_obj: Date object or string
        format_str: Desired format
    
    Returns:
        Formatted date string
    """
    if isinstance(date_obj, str):
        try:
            date_obj = datetime.strptime(date_obj, "%Y-%m-%d").date()
        except:
            return str(date_obj)
    
    if isinstance(date_obj, (date, datetime)):
        return date_obj.strftime(format_str)
    
    return str(date_obj)


def calculate_age(date_of_birth: str) -> int:
    """
    Calculate age from date of birth.
    
    Args:
        date_of_birth: DOB in YYYY-MM-DD format
    
    Returns:
        Age in years
    """
    try:
        dob = datetime.strptime(date_of_birth, "%Y-%m-%d").date()
        today = date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age
    except:
        return 0


# ===============================
# DATA VALIDATION
# ===============================
def validate_email(email: str) -> bool:
    """
    Simple email validation.
    
    Args:
        email: Email address
    
    Returns:
        True if valid, False otherwise
    """
    import re
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))


def validate_phone(phone: str) -> bool:
    """
    Simple phone validation.
    
    Args:
        phone: Phone number
    
    Returns:
        True if valid, False otherwise
    """
    import re
    # Remove all non-numeric characters
    phone_clean = re.sub(r'\D', '', phone)
    # Should be 10 or 11 digits
    return len(phone_clean) in [10, 11]


# ===============================
# MEDICAL CALCULATIONS
# ===============================
def calculate_bmi(weight_lbs: float, height_inches: float) -> float:
    """
    Calculate BMI from weight and height.
    
    Args:
        weight_lbs: Weight in pounds
        height_inches: Height in inches
    
    Returns:
        BMI value
    """
    # Convert to metric
    weight_kg = weight_lbs * 0.453592
    height_m = height_inches * 0.0254
    
    # Calculate BMI
    if height_m > 0:
        return round(weight_kg / (height_m ** 2), 1)
    return 0.0


def get_bmi_category(bmi: float) -> str:
    """
    Get BMI category.
    
    Args:
        bmi: BMI value
    
    Returns:
        Category string
    """
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def assess_blood_pressure(systolic: int, diastolic: int) -> Dict[str, str]:
    """
    Assess blood pressure reading.
    
    Args:
        systolic: Systolic BP
        diastolic: Diastolic BP
    
    Returns:
        Dictionary with category and recommendation
    """
    if systolic < 120 and diastolic < 80:
        return {
            "category": "Normal",
            "recommendation": "Maintain healthy lifestyle"
        }
    elif 120 <= systolic < 130 and diastolic < 80:
        return {
            "category": "Elevated",
            "recommendation": "Monitor regularly, lifestyle modifications"
        }
    elif 130 <= systolic < 140 or 80 <= diastolic < 90:
        return {
            "category": "Hypertension Stage 1",
            "recommendation": "Consult physician, lifestyle changes, possible medication"
        }
    elif systolic >= 140 or diastolic >= 90:
        return {
            "category": "Hypertension Stage 2",
            "recommendation": "Consult physician immediately, medication likely needed"
        }
    else:
        return {
            "category": "Crisis",
            "recommendation": "⚠️ Seek emergency medical care"
        }


# ===============================
# DATA FORMATTING
# ===============================
def format_vitals_display(vitals: Dict[str, Any]) -> str:
    """
    Format vital signs for display.
    
    Args:
        vitals: Vitals dictionary
    
    Returns:
        Formatted string
    """
    bp = f"{vitals.get('blood_pressure_systolic', 'N/A')}/{vitals.get('blood_pressure_diastolic', 'N/A')}"
    hr = vitals.get('heart_rate', 'N/A')
    temp = vitals.get('temperature', 'N/A')
    o2 = vitals.get('oxygen_saturation', 'N/A')
    
    return f"BP: {bp} mmHg | HR: {hr} BPM | Temp: {temp}°F | SpO2: {o2}%"


def format_medication_display(prescription: Dict[str, Any]) -> str:
    """
    Format prescription for display.
    
    Args:
        prescription: Prescription dictionary
    
    Returns:
        Formatted string
    """
    med_name = prescription.get('medication_name', 'Unknown')
    dosage = prescription.get('dosage', '')
    frequency = prescription.get('frequency', '')
    
    return f"{med_name} - {dosage}, {frequency}"


# ===============================
# JSON UTILITIES
# ===============================
def safe_json_loads(json_str: str) -> Any:
    """
    Safely parse JSON string.
    
    Args:
        json_str: JSON string
    
    Returns:
        Parsed object or None on error
    """
    try:
        return json.loads(json_str)
    except:
        return None


def safe_json_dumps(obj: Any, indent: int = 2) -> str:
    """
    Safely convert object to JSON string.
    
    Args:
        obj: Object to convert
        indent: Indentation level
    
    Returns:
        JSON string or error message
    """
    try:
        return json.dumps(obj, indent=indent, default=str)
    except Exception as e:
        return f"Error converting to JSON: {str(e)}"


# ===============================
# TEXT UTILITIES
# ===============================
def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def sanitize_input(text: str) -> str:
    """
    Sanitize user input.
    
    Args:
        text: Input text
    
    Returns:
        Sanitized text
    """
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove multiple spaces
    import re
    text = re.sub(r'\s+', ' ', text)
    
    return text


# ===============================
# LIST UTILITIES
# ===============================
def paginate_list(items: List[Any], page: int = 1, per_page: int = 10) -> Dict[str, Any]:
    """
    Paginate a list of items.
    
    Args:
        items: List of items
        page: Page number (1-indexed)
        per_page: Items per page
    
    Returns:
        Dictionary with paginated results
    """
    total_items = len(items)
    total_pages = (total_items + per_page - 1) // per_page
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    return {
        "items": items[start_idx:end_idx],
        "page": page,
        "per_page": per_page,
        "total_items": total_items,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1
    }


# ===============================
# STATUS INDICATORS
# ===============================
def get_status_icon(status: str) -> str:
    """
    Get icon for status.
    
    Args:
        status: Status string
    
    Returns:
        Emoji icon
    """
    status_icons = {
        "active": "✅",
        "completed": "✔️",
        "cancelled": "❌",
        "scheduled": "📅",
        "pending": "⏳",
        "critical": "🔴",
        "warning": "⚠️",
        "normal": "🟢",
        "abnormal": "🟡"
    }
    
    return status_icons.get(status.lower(), "⚪")


def get_priority_level(value: Any, thresholds: Dict[str, float]) -> str:
    """
    Determine priority level based on thresholds.
    
    Args:
        value: Value to check
        thresholds: Dictionary with 'low', 'medium', 'high' thresholds
    
    Returns:
        Priority level string
    """
    if value < thresholds.get('low', 0):
        return "Low"
    elif value < thresholds.get('medium', 50):
        return "Medium"
    elif value < thresholds.get('high', 100):
        return "High"
    else:
        return "Critical"