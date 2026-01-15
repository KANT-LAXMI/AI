import streamlit as st
import requests
from rag_utils import load_knowledge_base, build_faiss_index, retrieve_docs

# -------------------------------
# Config
# -------------------------------
API_URL = "http://127.0.0.1:8000/complaints"

# Load RAG knowledge base
docs, ids = load_knowledge_base()
index, embeddings = build_faiss_index(docs)

# -------------------------------
# Page Layout
# -------------------------------
st.set_page_config(page_title="Customer Service Chatbot", layout="wide")

st.markdown("""
<style>
/* Customize background and text */
body {
    background-color: #f5f7fa;
}
h1 {
    color: #2c3e50;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    height: 40px;
    width: 100%;
}
.stTextInput>div>div>input {
    height: 35px;
}
.stTextArea>div>div>textarea {
    height: 80px;
}
</style>
""", unsafe_allow_html=True)

st.title("💬 Customer Service Chatbot")

st.markdown("Ask any question or file/retrieve complaints seamlessly!")

# -------------------------------
# Chat / Complaint Input
# -------------------------------
user_input = st.text_input("Your Message", placeholder="E.g., I want to file a complaint or How can I get a refund?")

if user_input:
    # -------------------------------
    # 1️⃣ Retrieve Complaint Details
    # -------------------------------
    if "show details for complaint" in user_input.lower():
        words = user_input.split()
        complaint_id = words[-1]
        response = requests.get(f"{API_URL}/{complaint_id}")
        if response.status_code == 200:
            data = response.json()
            st.markdown("### 📄 Complaint Details")
            st.markdown(f"**Complaint ID:** {data['complaint_id']}")
            st.markdown(f"**Name:** {data['name']}")
            st.markdown(f"**Phone:** {data['phone_number']}")
            st.markdown(f"**Email:** {data['email']}")
            st.markdown(f"**Details:** {data['complaint_details']}")
            st.markdown(f"**Created At:** {data['created_at']}")
        else:
            st.error("❌ Complaint not found")

    # -------------------------------
    # 2️⃣ File a Complaint
    # -------------------------------
    elif "file a complaint" in user_input.lower():
        st.markdown("### 📝 File a New Complaint")
        with st.form(key="complaint_form"):
            name = st.text_input("Name")
            phone = st.text_input("Phone Number")
            email = st.text_input("Email")
            details = st.text_area("Complaint Details")
            submit_button = st.form_submit_button(label="Submit Complaint")
        
        if submit_button:
            if not name or not phone or not email or not details:
                st.warning("⚠️ Please fill all fields before submitting.")
            else:
                payload = {
                    "name": name,
                    "phone_number": phone,
                    "email": email,
                    "complaint_details": details
                }
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✅ Complaint created successfully! Your ID is: **{data['complaint_id']}**")
                else:
                    st.error("❌ Failed to create complaint")

    # -------------------------------
    # 3️⃣ Normal Query → RAG Response
    # -------------------------------
    else:
        st.markdown("### 🤖 Chatbot Answer (Knowledge Base)")
        results = retrieve_docs(user_input, docs, index, embeddings)
        for i, r in enumerate(results, start=1):
            st.markdown(f"**Result {i}:** {r}")
