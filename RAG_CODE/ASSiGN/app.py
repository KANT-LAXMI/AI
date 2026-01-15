import streamlit as st
import requests
from rag_utils import (load_knowledge_base, build_faiss_index, retrieve_docs, 
                        extract_complaint_id, is_complaint_query, is_retrieval_query)
from datetime import datetime
import time

# Configuration
API_URL = "http://127.0.0.1:8000/complaints"

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! 👋 I'm your customer service assistant. I can help you with:\n\n• Filing new complaints\n• Checking complaint status\n• Answering questions about our policies\n\nHow can I assist you today?",
            "timestamp": datetime.now()
        }
    ]

if 'conversation_state' not in st.session_state:
    st.session_state.conversation_state = {
        'stage': 'idle',
        'complaint_data': {}
    }

# Load RAG components
@st.cache_resource
def load_rag_components():
    docs, titles, ids = load_knowledge_base()
    index, embeddings = build_faiss_index(docs)
    return docs, titles, ids, index, embeddings

docs, titles, ids, index, embeddings = load_rag_components()

# Page Config
st.set_page_config(
    page_title="Customer Service Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stChatMessage {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #e3f2fd !important;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5 !important;
        margin-right: 20%;
    }
    .stButton>button {
        background-color: #1976d2;
        color: white;
        border-radius: 20px;
        padding: 10px 24px;
        font-weight: 500;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1565c0;
    }
    .chat-input {
        position: fixed;
        bottom: 0;
        background-color: white;
        padding: 20px;
        border-top: 1px solid #ddd;
    }
    h1 {
        color: #1976d2;
        font-weight: 600;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1976d2;
        margin: 10px 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("💬 Chatbot Info")
    st.markdown("""
    ### Features
    - 🤖 AI-powered responses
    - 📝 File complaints easily
    - 🔍 Check complaint status
    - 📚 Knowledge base search
    
    ### Quick Actions
    """)
    
    if st.button("📝 File New Complaint"):
        st.session_state.messages.append({
            "role": "assistant",
            "content": "I'll help you file a complaint. First, may I have your full name?",
            "timestamp": datetime.now()
        })
        st.session_state.conversation_state['stage'] = 'awaiting_name'
        st.rerun()
    
    if st.button("🔍 Check Complaint Status"):
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Please provide your complaint ID to check the status.",
            "timestamp": datetime.now()
        })
        st.rerun()
    
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared! How can I help you today?",
                "timestamp": datetime.now()
            }
        ]
        st.session_state.conversation_state = {'stage': 'idle', 'complaint_data': {}}
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.info("This chatbot uses RAG (Retrieval-Augmented Generation) to provide accurate information from our knowledge base.")

# Main Chat Interface
st.title("💬 Customer Service Chatbot")
st.markdown("---")

# Chat messages container
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            st.caption(message["timestamp"].strftime("%I:%M %p"))

# Chat input
user_input = st.chat_input("Type your message here...")

def add_message(role, content):
    """Add message to chat history"""
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now()
    })

def process_knowledge_query(query):
    """Process general knowledge base queries using RAG"""
    results = retrieve_docs(query, docs, titles, index, top_k=2)
    
    if results:
        response = "Based on our knowledge base, here's what I found:\n\n"
        for idx, (title, content, score) in enumerate(results, 1):
            if score < 1.5:  # Relevance threshold
                response += f"**{title}:**\n{content}\n\n"
        
        if "Based on our knowledge base, here's what I found:\n\n" == response:
            response = "I couldn't find specific information about that in our knowledge base. However, you can:\n\n"
            response += "• File a complaint if you're experiencing an issue\n"
            response += "• Contact support at support@company.com\n"
            response += "• Call our helpline at +1-800-555-0199"
    else:
        response = "I couldn't find relevant information. Can you rephrase your question or contact support?"
    
    return response

def handle_complaint_flow(user_input):
    """Handle the multi-step complaint filing process"""
    state = st.session_state.conversation_state
    stage = state['stage']
    
    if stage == 'idle' or stage == 'awaiting_name':
        state['complaint_data']['name'] = user_input
        state['stage'] = 'awaiting_phone'
        return f"Thank you, {user_input}! What's your phone number?"
    
    elif stage == 'awaiting_phone':
        state['complaint_data']['phone_number'] = user_input
        state['stage'] = 'awaiting_email'
        return "Got it! Please provide your email address."
    
    elif stage == 'awaiting_email':
        state['complaint_data']['email'] = user_input
        state['stage'] = 'awaiting_details'
        return "Thanks! Now, please describe your complaint in detail."
    
    elif stage == 'awaiting_details':
        state['complaint_data']['complaint_details'] = user_input
        
        # Create complaint via API
        try:
            response = requests.post(API_URL, json=state['complaint_data'], timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                complaint_id = data['complaint_id']
                
                # Reset state
                state['stage'] = 'idle'
                state['complaint_data'] = {}
                
                return f"""✅ **Complaint Successfully Filed!**

**Complaint ID:** `{complaint_id}`

Please save this ID for future reference. You can check your complaint status anytime by providing this ID.

**What happens next:**
• You'll receive an acknowledgment email within 24 hours
• Our team will review your complaint
• Resolution typically takes 3-5 business days

Is there anything else I can help you with?"""
            else:
                state['stage'] = 'idle'
                return "❌ Sorry, there was an error creating your complaint. Please try again or contact support."
        
        except requests.exceptions.RequestException as e:
            state['stage'] = 'idle'
            return "❌ Unable to connect to the complaint system. Please ensure the API is running (uvicorn api:app --reload) and try again."

def handle_complaint_retrieval(complaint_id):
    """Retrieve and display complaint details"""
    try:
        response = requests.get(f"{API_URL}/{complaint_id}", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return f"""📋 **Complaint Details**

**Complaint ID:** `{data['complaint_id']}`
**Name:** {data['name']}
**Phone:** {data['phone_number']}
**Email:** {data['email']}
**Details:** {data['complaint_details']}
**Filed On:** {data['created_at']}

**Status:** Under Review ⏳

Is there anything else you'd like to know?"""
        else:
            return "❌ Complaint not found. Please check the ID and try again."
    
    except requests.exceptions.RequestException:
        return "❌ Unable to retrieve complaint. Please ensure the API is running."

# Process user input
if user_input:
    # Add user message
    add_message("user", user_input)
    
    # Show typing indicator
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            time.sleep(0.5)
            
            state = st.session_state.conversation_state
            
            # Check if in complaint flow
            if state['stage'] != 'idle':
                response = handle_complaint_flow(user_input)
            
            # Check for complaint ID retrieval
            elif is_retrieval_query(user_input):
                complaint_id = extract_complaint_id(user_input)
                if complaint_id:
                    response = handle_complaint_retrieval(complaint_id)
                else:
                    response = "Please provide a valid complaint ID (format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)"
            
            # Check if user wants to file complaint
            elif is_complaint_query(user_input):
                st.session_state.conversation_state['stage'] = 'awaiting_name'
                response = "I'm sorry to hear you're experiencing an issue. I'll help you file a complaint.\n\nFirst, may I have your full name?"
            
            # General knowledge query
            else:
                response = process_knowledge_query(user_input)
            
            # Add bot response
            add_message("assistant", response)
            st.rerun()