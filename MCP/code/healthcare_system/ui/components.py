"""
UI Components
Reusable Streamlit components for the healthcare system
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime


# ===============================
# PATIENT DISPLAY COMPONENTS
# ===============================
def display_patient_card(patient: Dict[str, Any]):
    """
    Display a patient information card.
    
    Args:
        patient: Patient dictionary from database
    """
    with st.container():
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; margin-bottom: 20px;">
            <h3>👤 {patient['first_name']} {patient['last_name']}</h3>
            <p><strong>Patient ID:</strong> {patient['patient_id']}</p>
            <p><strong>DOB:</strong> {patient['date_of_birth']} | <strong>Gender:</strong> {patient['gender']}</p>
            <p><strong>Blood Type:</strong> {patient.get('blood_type', 'N/A')} | <strong>Phone:</strong> {patient.get('phone', 'N/A')}</p>
            <p><strong>Email:</strong> {patient.get('email', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)


def display_patient_summary(summary: Dict[str, Any]):
    """
    Display patient summary statistics.
    
    Args:
        summary: Summary dictionary from database
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", summary['total_records'])
    with col2:
        st.metric("Appointments", summary['total_appointments'])
    with col3:
        st.metric("Active Prescriptions", summary['active_prescriptions'])
    with col4:
        if summary['last_visit']:
            st.metric("Last Visit", summary['last_visit'])
        else:
            st.metric("Last Visit", "N/A")


# ===============================
# MEDICAL RECORDS COMPONENTS
# ===============================
def display_medical_records(records: List[Dict[str, Any]]):
    """
    Display medical records in a formatted timeline.
    
    Args:
        records: List of medical record dictionaries
    """
    if not records:
        st.info("No medical records found")
        return
    
    for record in records:
        with st.expander(f"📋 {record['visit_date']} - {record['diagnosis']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Diagnosis:**", record['diagnosis'])
                st.write("**Doctor:**", record.get('doctor_name', 'N/A'))
                st.write("**Symptoms:**", record.get('symptoms', 'N/A'))
            
            with col2:
                st.write("**Treatment:**", record.get('treatment', 'N/A'))
                st.write("**Prescription:**", record.get('prescription', 'N/A'))
                if record.get('notes'):
                    st.write("**Notes:**", record['notes'])


# ===============================
# APPOINTMENT COMPONENTS
# ===============================
def display_appointments(appointments: List[Dict[str, Any]]):
    """
    Display appointments in a table format.
    
    Args:
        appointments: List of appointment dictionaries
    """
    if not appointments:
        st.info("No appointments found")
        return
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(appointments)
    
    # Select and rename columns
    display_columns = {
        'appointment_id': 'ID',
        'first_name': 'First Name',
        'last_name': 'Last Name',
        'appointment_date': 'Date',
        'appointment_type': 'Type',
        'doctor_name': 'Doctor',
        'status': 'Status'
    }
    
    # Filter columns that exist
    existing_cols = [col for col in display_columns.keys() if col in df.columns]
    df_display = df[existing_cols].rename(columns=display_columns)
    
    # Apply color coding to status
    def color_status(val):
        if val == 'Scheduled':
            return 'background-color: #90EE90'
        elif val == 'Completed':
            return 'background-color: #87CEEB'
        elif val == 'Cancelled':
            return 'background-color: #FFB6C1'
        return ''
    
    if 'Status' in df_display.columns:
        styled_df = df_display.style.applymap(color_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.dataframe(df_display, use_container_width=True)


# ===============================
# PRESCRIPTION COMPONENTS
# ===============================
def display_prescriptions(prescriptions: List[Dict[str, Any]]):
    """
    Display prescriptions with medication details.
    
    Args:
        prescriptions: List of prescription dictionaries
    """
    if not prescriptions:
        st.info("No prescriptions found")
        return
    
    for rx in prescriptions:
        status_icon = "✅" if rx['status'] == 'Active' else "⏹️"
        
        with st.expander(f"{status_icon} {rx['medication_name']} - {rx['dosage']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Generic Name:**", rx.get('generic_name', 'N/A'))
                st.write("**Dosage:**", rx['dosage'])
                st.write("**Frequency:**", rx['frequency'])
                st.write("**Duration:**", rx.get('duration', 'N/A'))
            
            with col2:
                st.write("**Prescribed Date:**", rx['prescribed_date'])
                st.write("**Status:**", rx['status'])
                if rx.get('side_effects'):
                    st.write("**Side Effects:**", rx['side_effects'])


# ===============================
# VITALS VISUALIZATION
# ===============================
def plot_vitals_chart(vitals: List[Dict[str, Any]], parameter: str):
    """
    Create a line chart for vital signs over time.
    
    Args:
        vitals: List of vitals dictionaries
        parameter: Which parameter to plot (e.g., 'heart_rate', 'temperature')
    """
    if not vitals:
        st.info("No vitals data available")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(vitals)
    
    # Check if parameter exists
    if parameter not in df.columns:
        st.warning(f"No data for {parameter}")
        return
    
    # Remove None values
    df = df[df[parameter].notna()]
    
    if df.empty:
        st.warning(f"No valid data for {parameter}")
        return
    
    # Create plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['measurement_date'],
        y=df[parameter],
        mode='lines+markers',
        name=parameter.replace('_', ' ').title(),
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8)
    ))
    
    # Add normal range bands (example for heart rate)
    if parameter == 'heart_rate':
        fig.add_hrect(y0=60, y1=100, fillcolor="green", opacity=0.1, 
                     annotation_text="Normal Range", annotation_position="top left")
    
    fig.update_layout(
        title=f"{parameter.replace('_', ' ').title()} Over Time",
        xaxis_title="Date",
        yaxis_title=parameter.replace('_', ' ').title(),
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_vitals_table(vitals: List[Dict[str, Any]]):
    """
    Display vitals in a table format.
    
    Args:
        vitals: List of vitals dictionaries
    """
    if not vitals:
        st.info("No vitals recorded")
        return
    
    df = pd.DataFrame(vitals)
    
    # Select relevant columns
    display_cols = [
        'measurement_date',
        'blood_pressure_systolic',
        'blood_pressure_diastolic',
        'heart_rate',
        'temperature',
        'oxygen_saturation',
        'weight'
    ]
    
    # Filter existing columns
    existing_cols = [col for col in display_cols if col in df.columns]
    df_display = df[existing_cols]
    
    # Rename for display
    df_display.columns = [col.replace('_', ' ').title() for col in df_display.columns]
    
    st.dataframe(df_display, use_container_width=True)


# ===============================
# STATISTICS DASHBOARD
# ===============================
def display_statistics_dashboard(stats: Dict[str, Any]):
    """
    Display system statistics dashboard.
    
    Args:
        stats: Statistics dictionary from database
    """
    st.subheader("📊 System Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Patients",
            value=stats.get('total_patients', 0),
            delta=None
        )
    
    with col2:
        st.metric(
            label="Total Appointments",
            value=stats.get('total_appointments', 0),
            delta=None
        )
    
    with col3:
        st.metric(
            label="Today's Appointments",
            value=stats.get('todays_appointments', 0),
            delta=None
        )
    
    with col4:
        st.metric(
            label="Active Prescriptions",
            value=stats.get('active_prescriptions', 0),
            delta=None
        )


# ===============================
# AI RESPONSE DISPLAY
# ===============================
def display_ai_response(response: Dict[str, Any]):
    """
    Display AI agent response with formatting.
    
    Args:
        response: Response dictionary from healthcare agent
    """
    if response['status'] == 'success':
        st.success("✅ Query completed successfully")
        
        # Display answer
        st.markdown("### 🤖 AI Response:")
        st.write(response['answer'])
        
        # Display tools used
        if response.get('tool_calls'):
            with st.expander("🔧 Tools Used"):
                for i, tool_call in enumerate(response['tool_calls'], 1):
                    st.write(f"{i}. **{tool_call['tool']}**")
                    if tool_call.get('args'):
                        st.json(tool_call['args'])
    else:
        st.error(f"❌ Error: {response.get('error', 'Unknown error')}")


# ===============================
# FORM COMPONENTS
# ===============================
def create_patient_form():
    """Create a form for adding a new patient"""
    with st.form("new_patient_form"):
        st.subheader("Add New Patient")
        
        col1, col2 = st.columns(2)
        
        with col1:
            first_name = st.text_input("First Name*")
            last_name = st.text_input("Last Name*")
            dob = st.date_input("Date of Birth*")
            gender = st.selectbox("Gender*", ["Male", "Female", "Other"])
            blood_type = st.selectbox("Blood Type", ["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
        
        with col2:
            phone = st.text_input("Phone")
            email = st.text_input("Email*")
            address = st.text_area("Address")
            emergency_contact = st.text_input("Emergency Contact")
            emergency_phone = st.text_input("Emergency Phone")
        
        insurance_id = st.text_input("Insurance ID")
        
        submitted = st.form_submit_button("Add Patient")
        
        if submitted:
            if not all([first_name, last_name, email]):
                st.error("Please fill in all required fields (*)")
                return None
            
            return {
                'first_name': first_name,
                'last_name': last_name,
                'date_of_birth': str(dob),
                'gender': gender,
                'blood_type': blood_type if blood_type else None,
                'phone': phone,
                'email': email,
                'address': address,
                'emergency_contact': emergency_contact,
                'emergency_phone': emergency_phone,
                'insurance_id': insurance_id
            }
    
    return None