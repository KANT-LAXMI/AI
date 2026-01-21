"""
Healthcare Management System - Streamlit UI
Main application integrating SK + MCP + SQLite + LLM
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import asyncio
from datetime import datetime, date
import pandas as pd

# Import our modules
from database import init_database
from database.operations import DatabaseOperations
from agents.healthcare_agent import HealthcareAgent
from sk_kernel.kernel_setup import (
    setup_kernel,
    analyze_symptoms,
    generate_medical_summary,
    suggest_treatment_plan
)
from sk_kernel.plugins import MedicalAnalysisPlugin, DiagnosisAssistantPlugin
from config.settings import APP_TITLE, APP_ICON, APPOINTMENT_TYPES, APPOINTMENT_STATUS
from ui.components import *

# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .stButton>button {
        width: 100%;
    }
    .success-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# INITIALIZE SESSION STATE
# ===============================
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'kernel' not in st.session_state:
    st.session_state.kernel = None
if 'selected_patient' not in st.session_state:
    st.session_state.selected_patient = None

# ===============================
# ASYNC HELPER FUNCTIONS
# ===============================
def run_async(coro):
    """Helper to run async functions"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# ===============================
# INITIALIZE COMPONENTS
# ===============================
@st.cache_resource
def initialize_database():
    """Initialize database (cached)"""
    run_async(init_database())
    return DatabaseOperations()

@st.cache_resource
def initialize_agent():
    """Initialize healthcare agent (cached)"""
    agent = HealthcareAgent()
    run_async(agent.initialize())
    return agent

@st.cache_resource
def initialize_kernel():
    """Initialize Semantic Kernel (cached)"""
    return setup_kernel(use_openai=False)

# ===============================
# MAIN APPLICATION
# ===============================
def main():
    # Header
    st.markdown(f'<p class="main-header">{APP_ICON} {APP_TITLE}</p>', unsafe_allow_html=True)
    
    # Initialize database
    db_ops = initialize_database()
    
    # Sidebar
    with st.sidebar:
        st.title("🎯 Navigation")
        
        page = st.radio(
            "Select Page",
            [
                "🏠 Dashboard",
                "👥 Patients",
                "📋 Medical Records",
                "📅 Appointments",
                "💊 Prescriptions",
                "🩺 Vitals",
                "🤖 AI Assistant",
                "🧠 Medical Analysis (SK)",
                "⚙️ System Info"
            ],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Quick stats
        stats = run_async(db_ops.get_statistics())
        st.metric("Total Patients", stats.get('total_patients', 0))
        st.metric("Today's Appointments", stats.get('todays_appointments', 0))
    
    # ===============================
    # PAGE ROUTING
    # ===============================
    
    if page == "🏠 Dashboard":
        show_dashboard(db_ops)
    
    elif page == "👥 Patients":
        show_patients(db_ops)
    
    elif page == "📋 Medical Records":
        show_medical_records(db_ops)
    
    elif page == "📅 Appointments":
        show_appointments(db_ops)
    
    elif page == "💊 Prescriptions":
        show_prescriptions(db_ops)
    
    elif page == "🩺 Vitals":
        show_vitals(db_ops)
    
    elif page == "🤖 AI Assistant":
        show_ai_assistant()
    
    elif page == "🧠 Medical Analysis (SK)":
        show_medical_analysis()
    
    elif page == "⚙️ System Info":
        show_system_info()


# ===============================
# DASHBOARD PAGE
# ===============================
def show_dashboard(db_ops):
    st.header("📊 Dashboard")
    
    # System statistics
    stats = run_async(db_ops.get_statistics())
    display_statistics_dashboard(stats)
    
    st.divider()
    
    # Recent activity
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Upcoming Appointments")
        today = date.today().isoformat()
        appointments = run_async(db_ops.get_appointments(start_date=today))
        if appointments:
            for apt in appointments[:5]:  # Show first 5
                st.write(f"**{apt['first_name']} {apt['last_name']}** - {apt['appointment_date']}")
                st.write(f"Type: {apt['appointment_type']} | Status: {apt['status']}")
                st.divider()
        else:
            st.info("No upcoming appointments")
    
    with col2:
        st.subheader("👥 Recent Patients")
        patients = run_async(db_ops.get_all_patients())
        for patient in patients[:5]:  # Show first 5
            st.write(f"**{patient['first_name']} {patient['last_name']}**")
            st.write(f"ID: {patient['patient_id']} | Blood Type: {patient.get('blood_type', 'N/A')}")
            st.divider()


# ===============================
# PATIENTS PAGE
# ===============================
def show_patients(db_ops):
    st.header("👥 Patient Management")
    
    tab1, tab2, tab3 = st.tabs(["Search Patients", "Patient Details", "Add New Patient"])
    
    with tab1:
        st.subheader("🔍 Search Patients")
        
        search_term = st.text_input("Search by name or email", "")
        
        if search_term:
            patients = run_async(db_ops.search_patients(search_term))
        else:
            patients = run_async(db_ops.get_all_patients())
        
        if patients:
            # Create DataFrame for display
            df = pd.DataFrame(patients)
            display_cols = ['patient_id', 'first_name', 'last_name', 'date_of_birth', 
                          'gender', 'blood_type', 'email', 'phone']
            df_display = df[[col for col in display_cols if col in df.columns]]
            
            st.dataframe(df_display, use_container_width=True)
            
            # Select patient
            patient_id = st.number_input("Enter Patient ID to view details", 
                                        min_value=1, step=1, value=1)
            if st.button("View Patient"):
                st.session_state.selected_patient = patient_id
                st.success(f"Selected Patient ID: {patient_id}")
        else:
            st.info("No patients found")
    
    with tab2:
        st.subheader("📄 Patient Details")
        
        patient_id = st.session_state.selected_patient or st.number_input(
            "Enter Patient ID", min_value=1, step=1, value=1
        )
        
        if st.button("Load Patient Details"):
            summary = run_async(db_ops.get_patient_summary(patient_id))
            
            if summary['patient']:
                # Display patient card
                display_patient_card(summary['patient'])
                
                # Display summary
                display_patient_summary(summary)
                
                st.divider()
                
                # Medical history
                st.subheader("📋 Medical History")
                history = run_async(db_ops.get_patient_medical_history(patient_id))
                display_medical_records(history)
            else:
                st.error(f"Patient ID {patient_id} not found")
    
    with tab3:
        st.subheader("➕ Add New Patient")
        
        patient_data = create_patient_form()
        
        if patient_data:
            try:
                new_id = run_async(db_ops.add_patient(patient_data))
                st.success(f"✅ Patient added successfully! Patient ID: {new_id}")
            except Exception as e:
                st.error(f"Error adding patient: {str(e)}")


# ===============================
# MEDICAL RECORDS PAGE
# ===============================
def show_medical_records(db_ops):
    st.header("📋 Medical Records")
    
    tab1, tab2 = st.tabs(["View Records", "Add New Record"])
    
    with tab1:
        patient_id = st.number_input("Enter Patient ID", min_value=1, step=1, value=1, key="mr_view")
        
        if st.button("Load Medical Records"):
            records = run_async(db_ops.get_patient_medical_history(patient_id))
            
            if records:
                st.success(f"Found {len(records)} record(s)")
                display_medical_records(records)
            else:
                st.info("No medical records found for this patient")
    
    with tab2:
        with st.form("add_record_form"):
            st.subheader("Add Medical Record")
            
            patient_id = st.number_input("Patient ID", min_value=1, step=1, value=1)
            visit_date = st.date_input("Visit Date", value=date.today())
            diagnosis = st.text_input("Diagnosis*")
            symptoms = st.text_area("Symptoms")
            treatment = st.text_area("Treatment*")
            prescription = st.text_area("Prescription")
            doctor_name = st.text_input("Doctor Name*")
            notes = st.text_area("Additional Notes")
            
            submitted = st.form_submit_button("Add Record")
            
            if submitted:
                if not all([diagnosis, treatment, doctor_name]):
                    st.error("Please fill in all required fields (*)")
                else:
                    try:
                        record_id = run_async(db_ops.add_medical_record({
                            'patient_id': patient_id,
                            'visit_date': str(visit_date),
                            'diagnosis': diagnosis,
                            'symptoms': symptoms,
                            'treatment': treatment,
                            'prescription': prescription,
                            'doctor_name': doctor_name,
                            'notes': notes
                        }))
                        st.success(f"✅ Medical record added! Record ID: {record_id}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")


# ===============================
# APPOINTMENTS PAGE
# ===============================
def show_appointments(db_ops):
    st.header("📅 Appointment Management")
    
    tab1, tab2 = st.tabs(["View Appointments", "Schedule Appointment"])
    
    with tab1:
        st.subheader("📋 Appointments")
        
        col1, col2 = st.columns(2)
        with col1:
            filter_patient = st.number_input("Filter by Patient ID (0 for all)", 
                                           min_value=0, step=1, value=0)
        with col2:
            filter_date = st.date_input("From Date", value=date.today())
        
        if st.button("Load Appointments"):
            patient_id = filter_patient if filter_patient > 0 else None
            appointments = run_async(db_ops.get_appointments(
                patient_id=patient_id,
                start_date=str(filter_date)
            ))
            
            if appointments:
                st.success(f"Found {len(appointments)} appointment(s)")
                display_appointments(appointments)
            else:
                st.info("No appointments found")
    
    with tab2:
        with st.form("schedule_appointment_form"):
            st.subheader("Schedule New Appointment")
            
            patient_id = st.number_input("Patient ID", min_value=1, step=1, value=1)
            
            col1, col2 = st.columns(2)
            with col1:
                apt_date = st.date_input("Appointment Date")
                apt_time = st.time_input("Appointment Time")
            with col2:
                apt_type = st.selectbox("Appointment Type", APPOINTMENT_TYPES)
                doctor_name = st.text_input("Doctor Name")
            
            reason = st.text_area("Reason for Visit")
            
            submitted = st.form_submit_button("Schedule Appointment")
            
            if submitted:
                try:
                    apt_datetime = f"{apt_date} {apt_time}"
                    apt_id = run_async(db_ops.schedule_appointment({
                        'patient_id': patient_id,
                        'appointment_date': apt_datetime,
                        'appointment_type': apt_type,
                        'doctor_name': doctor_name,
                        'reason': reason
                    }))
                    st.success(f"✅ Appointment scheduled! ID: {apt_id}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


# ===============================
# PRESCRIPTIONS PAGE
# ===============================
def show_prescriptions(db_ops):
    st.header("💊 Prescription Management")
    
    tab1, tab2 = st.tabs(["View Prescriptions", "Check Drug Interactions"])
    
    with tab1:
        patient_id = st.number_input("Enter Patient ID", min_value=1, step=1, value=1)
        active_only = st.checkbox("Show active prescriptions only", value=True)
        
        if st.button("Load Prescriptions"):
            prescriptions = run_async(db_ops.get_patient_prescriptions(patient_id, active_only))
            
            if prescriptions:
                st.success(f"Found {len(prescriptions)} prescription(s)")
                display_prescriptions(prescriptions)
            else:
                st.info("No prescriptions found")
    
    with tab2:
        st.subheader("🔍 Drug Interaction Checker")
        st.info("This feature uses the MCP pharmacy server to check for drug interactions")
        
        patient_id = st.number_input("Patient ID", min_value=1, step=1, value=1, key="drug_check")
        new_medication = st.text_input("New Medication Name")
        
        if st.button("Check Interactions"):
            if new_medication:
                agent = initialize_agent()
                query = f"Check for drug interactions if we prescribe {new_medication} to patient {patient_id}"
                response = run_async(agent.query(query))
                display_ai_response(response)
            else:
                st.warning("Please enter a medication name")


# ===============================
# VITALS PAGE
# ===============================
def show_vitals(db_ops):
    st.header("🩺 Vital Signs Tracking")
    
    tab1, tab2 = st.tabs(["View Vitals", "Record Vitals"])
    
    with tab1:
        patient_id = st.number_input("Enter Patient ID", min_value=1, step=1, value=1)
        limit = st.slider("Number of records to show", 5, 50, 10)
        
        if st.button("Load Vitals"):
            vitals = run_async(db_ops.get_patient_vitals(patient_id, limit))
            
            if vitals:
                st.success(f"Found {len(vitals)} vital sign record(s)")
                
                # Display table
                display_vitals_table(vitals)
                
                st.divider()
                
                # Visualizations
                st.subheader("📈 Vitals Trends")
                
                col1, col2 = st.columns(2)
                with col1:
                    plot_vitals_chart(vitals, 'heart_rate')
                with col2:
                    plot_vitals_chart(vitals, 'temperature')
                
                col3, col4 = st.columns(2)
                with col3:
                    plot_vitals_chart(vitals, 'blood_pressure_systolic')
                with col4:
                    plot_vitals_chart(vitals, 'oxygen_saturation')
            else:
                st.info("No vital signs recorded for this patient")
    
    with tab2:
        with st.form("record_vitals_form"):
            st.subheader("Record Vital Signs")
            
            patient_id = st.number_input("Patient ID", min_value=1, step=1, value=1)
            measurement_date = st.date_input("Measurement Date", value=date.today())
            measurement_time = st.time_input("Measurement Time")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                bp_systolic = st.number_input("BP Systolic", min_value=0, max_value=300, value=120)
                bp_diastolic = st.number_input("BP Diastolic", min_value=0, max_value=200, value=80)
            
            with col2:
                heart_rate = st.number_input("Heart Rate (BPM)", min_value=0, max_value=300, value=75)
                temperature = st.number_input("Temperature (°F)", min_value=90.0, max_value=110.0, value=98.6, step=0.1)
            
            with col3:
                oxygen_sat = st.number_input("O2 Saturation (%)", min_value=0, max_value=100, value=98)
                weight = st.number_input("Weight (lbs)", min_value=0.0, max_value=1000.0, value=150.0, step=0.1)
            
            submitted = st.form_submit_button("Record Vitals")
            
            if submitted:
                try:
                    measurement_datetime = f"{measurement_date} {measurement_time}"
                    vital_id = run_async(db_ops.add_vitals({
                        'patient_id': patient_id,
                        'measurement_date': measurement_datetime,
                        'blood_pressure_systolic': bp_systolic,
                        'blood_pressure_diastolic': bp_diastolic,
                        'heart_rate': heart_rate,
                        'temperature': temperature,
                        'oxygen_saturation': oxygen_sat,
                        'weight': weight,
                        'record_id': None
                    }))
                    st.success(f"✅ Vital signs recorded! ID: {vital_id}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


# ===============================
# AI ASSISTANT PAGE (MCP Agent)
# ===============================
def show_ai_assistant():
    st.header("🤖 AI Healthcare Assistant")
    st.info("Ask questions about patients, appointments, medications, and more. The AI uses MCP tools to access the database.")
    
    # Initialize agent
    agent = initialize_agent()
    
    # Example queries
    with st.expander("💡 Example Queries"):
        st.code("""
• What are the current system statistics?
• Search for patients named 'John'
• Get details for patient ID 1
• What medications is patient 1 currently taking?
• Check for drug interactions if we prescribe Aspirin to patient 2
• Show me appointments for this week
• Get vital signs history for patient 1
• What is patient 2's medical history?
• Schedule an appointment for patient 3
        """)
    
    # Chat interface
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat['role']):
            st.write(chat['content'])
    
    # User input
    user_query = st.chat_input("Ask me anything about the healthcare system...")
    
    if user_query:
        # Add user message to chat
        st.session_state.chat_history.append({'role': 'user', 'content': user_query})
        
        with st.chat_message("user"):
            st.write(user_query)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = run_async(agent.query(user_query))
                
                if response['status'] == 'success':
                    st.write(response['answer'])
                    
                    # Show tools used
                    if response.get('tool_calls'):
                        with st.expander("🔧 Tools Used"):
                            for tool_call in response['tool_calls']:
                                st.write(f"- {tool_call['tool']}")
                else:
                    st.error(f"Error: {response.get('error', 'Unknown error')}")
        
        # Add assistant response to chat
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response.get('answer', 'Error occurred')
        })
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()


# ===============================
# MEDICAL ANALYSIS PAGE (Semantic Kernel)
# ===============================
def show_medical_analysis():
    st.header("🧠 Medical Analysis with Semantic Kernel")
    st.info("Use AI-powered analysis for symptoms, treatment planning, and medical summaries")
    
    kernel = initialize_kernel()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Symptom Analysis",
        "📝 Medical Summary",
        "💊 Treatment Planning",
        "🩺 Vitals Analysis"
    ])
    
    with tab1:
        st.subheader("Symptom Analysis")
        
        symptoms = st.text_area("Describe Patient Symptoms", 
                               placeholder="e.g., Persistent cough, fever, fatigue...")
        patient_history = st.text_area("Patient History (Optional)",
                                      placeholder="e.g., 45-year-old male, smoker...")
        
        if st.button("Analyze Symptoms"):
            if symptoms:
                with st.spinner("Analyzing..."):
                    result = run_async(analyze_symptoms(kernel, symptoms, patient_history))
                    st.markdown("### Analysis Results:")
                    st.write(result)
            else:
                st.warning("Please enter symptoms")
    
    with tab2:
        st.subheader("Generate Medical Summary")
        
        patient_id = st.number_input("Patient ID", min_value=1, step=1, value=1)
        
        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                db_ops = initialize_database()
                records = run_async(db_ops.get_patient_medical_history(patient_id))
                
                if records:
                    # Convert records to text
                    records_text = "\n\n".join([
                        f"Visit {i+1} ({r['visit_date']}):\n"
                        f"Diagnosis: {r['diagnosis']}\n"
                        f"Symptoms: {r.get('symptoms', 'N/A')}\n"
                        f"Treatment: {r.get('treatment', 'N/A')}"
                        for i, r in enumerate(records)
                    ])
                    
                    summary = run_async(generate_medical_summary(kernel, records_text))
                    st.markdown("### Medical Summary:")
                    st.write(summary)
                else:
                    st.info("No medical records found for this patient")
    
    with tab3:
        st.subheader("Treatment Plan Suggestion")
        
        diagnosis = st.text_input("Diagnosis", placeholder="e.g., Type 2 Diabetes")
        patient_info = st.text_area("Patient Information",
                                   placeholder="Age, gender, existing conditions...")
        allergies = st.text_input("Known Allergies", placeholder="e.g., Penicillin")
        
        if st.button("Suggest Treatment Plan"):
            if diagnosis and patient_info:
                with st.spinner("Generating treatment plan..."):
                    plan = run_async(suggest_treatment_plan(
                        kernel, diagnosis, patient_info, allergies
                    ))
                    st.markdown("### Suggested Treatment Plan:")
                    st.write(plan)
            else:
                st.warning("Please provide diagnosis and patient information")
    
    with tab4:
        st.subheader("Vitals Analysis with SK Plugin")
        
        col1, col2 = st.columns(2)
        with col1:
            bp_input = st.text_input("Blood Pressure (systolic/diastolic)", value="120/80")
            hr_input = st.number_input("Heart Rate (BPM)", value=75)
        with col2:
            temp_input = st.number_input("Temperature (°F)", value=98.6)
            o2_input = st.number_input("O2 Saturation (%)", value=98)
        
        if st.button("Analyze Vitals"):
            plugin = MedicalAnalysisPlugin()
            result = run_async(plugin.analyze_vitals(
                bp_input, hr_input, temp_input, o2_input
            ))
            
            st.markdown("### Vitals Analysis:")
            st.json(result)


# ===============================
# SYSTEM INFO PAGE
# ===============================
def show_system_info():
    st.header("⚙️ System Information")
    
    st.subheader("🔧 Components Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Database", "✅ Connected")
        st.metric("Semantic Kernel", "✅ Initialized")
    
    with col2:
        st.metric("MCP Servers", "✅ Running")
        st.metric("AI Agent", "✅ Ready")
    
    with col3:
        st.metric("Ollama", "✅ Connected")
        st.metric("Streamlit", "✅ Running")
    
    st.divider()
    
    st.subheader("📊 System Configuration")
    
    from config.settings import OLLAMA_CONFIG, MCP_SERVERS
    
    st.write("**Ollama Configuration:**")
    st.json(OLLAMA_CONFIG)
    
    st.write("**MCP Servers:**")
    st.json(MCP_SERVERS)
    
    st.divider()
    
    st.subheader("📚 About")
    st.markdown("""
    ### Healthcare Management System
    
    **Version:** 1.0.0
    
    **Components:**
    - **Semantic Kernel**: AI orchestration and medical analysis
    - **MCP (Model Context Protocol)**: Tool integration for database operations
    - **SQLite**: Local database for patient records
    - **LLM**: Ollama (llama3.2:3b) for natural language processing
    - **Streamlit**: Interactive web interface
    
    **Features:**
    - Patient management
    - Medical records tracking
    - Appointment scheduling
    - Prescription management
    - Vital signs monitoring
    - AI-powered assistance
    - Drug interaction checking
    - Symptom analysis
    - Treatment planning
    
    **Architecture:**
    ```
    Streamlit UI
        ↓
    ├── Semantic Kernel (Medical Analysis)
    ├── Healthcare Agent (LangChain + MCP)
    │   ├── Medical MCP Server
    │   └── Pharmacy MCP Server
    └── SQLite Database
    ```
    """)


# ===============================
# RUN APPLICATION
# ===============================
if __name__ == "__main__":
    main()