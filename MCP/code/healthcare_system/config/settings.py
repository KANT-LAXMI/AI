"""
Configuration Settings for Healthcare System
Centralized configuration management for all components
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===============================
# BASE PATHS
# ===============================
BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "database"
SERVERS_DIR = BASE_DIR / "servers"

# ===============================
# DATABASE CONFIGURATION
# ===============================
DATABASE_PATH = os.getenv("DATABASE_PATH", str(DATABASE_DIR / "healthcare.db"))

# ===============================
# LLM CONFIGURATION
# ===============================
OLLAMA_CONFIG = {
    "model": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
    "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    "temperature": float(os.getenv("TEMPERATURE", "0.3")),
    "max_tokens": int(os.getenv("MAX_TOKENS", "1000"))
}

OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "model": os.getenv("OPENAI_MODEL", "gpt-4"),
    "temperature": float(os.getenv("TEMPERATURE", "0.3")),
    "max_tokens": int(os.getenv("MAX_TOKENS", "1000"))
}

# ===============================
# MCP SERVER CONFIGURATION
# ===============================
MCP_SERVERS = {
    "medical": {
        "command": "python",
        "args": [str(SERVERS_DIR / "medical_server.py")],
        "transport": "stdio",
    },
    "pharmacy": {
        "command": "python",
        "args": [str(SERVERS_DIR / "pharmacy_server.py")],
        "transport": "stdio",
    }
}

# ===============================
# API KEYS
# ===============================
FDA_API_KEY = os.getenv("FDA_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

# ===============================
# APPLICATION SETTINGS
# ===============================
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
APP_TITLE = "🏥 Healthcare Patient Management System"
APP_ICON = "🏥"

# ===============================
# MEDICAL CONSTANTS
# ===============================
BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
APPOINTMENT_TYPES = ["General Checkup", "Follow-up", "Emergency", "Consultation", "Surgery", "Vaccination"]
APPOINTMENT_STATUS = ["Scheduled", "Completed", "Cancelled", "No-Show"]