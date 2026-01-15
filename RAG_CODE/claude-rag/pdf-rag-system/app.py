"""
Advanced PDF RAG System - Streamlit UI
Multi-Modal Document Question Answering with Tables, Charts, and Images
"""

import streamlit as st
import os
from pathlib import Path
import time
from advanced_pdf_processor import AdvancedPDFProcessor, process_folder
from rag_engine import RAGEngine
import json

# Page config
st.set_page_config(
    page_title="Advanced PDF RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #145a8c;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .source-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = []

# Header
st.markdown('<div class="main-header">📚 Advanced PDF RAG System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Modal Document Intelligence: Text, Tables, Charts & Images</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/pdf.png", width=80)
    st.title("⚙️ Configuration")
    
    # Settings
    st.subheader("📂 Document Management")
    
    pdf_folder = st.text_input(
        "PDF Folder Path",
        value="data",
        help="Path to folder containing PDF files"
    )
    
    output_folder = st.text_input(
        "Output Folder",
        value="extracted_data",
        help="Where to save extracted content"
    )
    
    st.divider()
    
    # Process PDFs
    st.subheader("🔄 Processing")
    
    if st.button("📥 Process PDFs", key="process_btn"):
        if not os.path.exists(pdf_folder):
            st.error(f"❌ Folder '{pdf_folder}' not found!")
        else:
            with st.spinner("Processing PDFs... This may take a few minutes."):
                try:
                    # Process PDFs
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Extracting content from PDFs...")
                    progress_bar.progress(25)
                    
                    chunks = process_folder(pdf_folder, output_folder)
                    st.session_state.processed_docs = chunks
                    
                    status_text.text("Building search index...")
                    progress_bar.progress(50)
                    
                    # Build RAG index
                    engine = RAGEngine()
                    engine.build_index(chunks)
                    st.session_state.rag_engine = engine
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")
                    
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"✅ Processed {len(chunks)} chunks from {len(os.listdir(pdf_folder))} PDFs")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Load existing index
    if st.button("📂 Load Existing Index", key="load_btn"):
        index_path = "vector_store/faiss_index.bin"
        if os.path.exists(index_path):
            with st.spinner("Loading index..."):
                try:
                    engine = RAGEngine()
                    engine.load_index(index_path)
                    st.session_state.rag_engine = engine
                    st.success(f"✅ Loaded {len(engine.documents)} chunks")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.error(f"❌ Index not found at {index_path}")
    
    st.divider()
    
    # RAG Settings
    st.subheader("🎛️ RAG Settings")
    
    top_k = st.slider(
        "Number of results to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="More results provide better context but may include noise"
    )
    
    temperature = st.slider(
        "Response creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower = more factual, Higher = more creative"
    )
    
    st.divider()
    
    # Status
    st.subheader("📊 Status")
    if st.session_state.rag_engine:
        st.success("✅ System Ready")
        st.metric("Indexed Chunks", len(st.session_state.rag_engine.documents))
    else:
        st.warning("⚠️ Please process PDFs or load an index")
    
    st.divider()
    
    # Clear conversation
    if st.button("🗑️ Clear Conversation"):
        st.session_state.conversation_history = []
        st.rerun()

# Main content area
if st.session_state.rag_engine is None:
    # Welcome screen
    st.markdown("""
    <div class="info-box">
        <h3>👋 Welcome to Advanced PDF RAG System!</h3>
        <p>This system can analyze complex PDF documents containing:</p>
        <ul>
            <li>✅ Regular text and paragraphs</li>
            <li>✅ Tables with financial data</li>
            <li>✅ Charts and graphs (line, bar, pie)</li>
            <li>✅ Scanned documents (via OCR)</li>
            <li>✅ Images and diagrams</li>
        </ul>
        <p><strong>To get started:</strong></p>
        <ol>
            <li>Place your PDF files in the 'data' folder (or specify a custom path)</li>
            <li>Click "📥 Process PDFs" in the sidebar</li>
            <li>Start asking questions about your documents!</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>🎯</h2>
            <h3>Smart Extraction</h3>
            <p>Extracts text, tables, and images with high accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>🔍</h2>
            <h3>Semantic Search</h3>
            <p>Finds relevant content using AI embeddings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>🤖</h2>
            <h3>AI Answers</h3>
            <p>Generates accurate answers with source citations</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Q&A Interface
    st.subheader("💬 Ask Questions About Your Documents")
    
    # Example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - What was the dividend per share in 2022-23?
        - Show me all the financial data from the tables
        - What does the share price chart indicate?
        - Summarize the key findings from page 5
        - What are the highest and lowest prices shown in the graph?
        - Extract all dates and amounts from the document
        """)
    
    # Question input
    query = st.text_input(
        "Your Question:",
        placeholder="e.g., What was the total dividend in 2022-23?",
        key="query_input"
    )
    
    col1, col2 = st.columns([1, 5])
    
    with col1:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
    
    with col2:
        if st.button("📋 Show Document Stats", use_container_width=True):
            if st.session_state.rag_engine:
                docs = st.session_state.rag_engine.documents
                
                # Calculate stats
                total_words = sum(len(doc.split()) for doc in docs)
                avg_words = total_words / len(docs) if docs else 0
                
                # Count content types
                content_types = {'text': 0, 'tables': 0, 'images': 0, 'charts': 0}
                for doc in docs:
                    if 'Table' in doc:
                        content_types['tables'] += 1
                    if any(word in doc for word in ['chart', 'graph', 'Chart', 'Graph']):
                        content_types['charts'] += 1
                    if any(word in doc for word in ['image', 'Image', 'Visual']):
                        content_types['images'] += 1
                    content_types['text'] += 1
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Chunks", len(docs))
                col2.metric("Total Words", f"{total_words:,}")
                col3.metric("Tables Found", content_types['tables'])
                col4.metric("Charts/Images", content_types['charts'] + content_types['images'])
    
    # Process question
    if ask_button and query:
        with st.spinner("🤔 Thinking..."):
            try:
                # Get answer
                answer, retrieved_docs = st.session_state.rag_engine.answer_question(
                    query,
                    top_k=top_k,
                    conversation_history=st.session_state.conversation_history
                )
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.markdown(f"<div class='success-box'>{answer}</div>", unsafe_allow_html=True)
                
                # Show sources
                with st.expander("📚 View Source Documents", expanded=False):
                    for i, doc in enumerate(retrieved_docs, 1):
                        # Extract metadata
                        if "[Document:" in doc:
                            metadata_end = doc.find("]") + 1
                            metadata = doc[:metadata_end]
                            content = doc[metadata_end:].strip()
                        else:
                            metadata = "Unknown source"
                            content = doc
                        
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>📄 Source {i}</strong><br>
                            <em>{metadata}</em><br>
                            <p style="margin-top: 0.5rem; font-size: 0.9rem;">{content[:300]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Update conversation history
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": query
                })
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": answer
                })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Show conversation history
    if st.session_state.conversation_history:
        st.divider()
        st.subheader("📜 Conversation History")
        
        for i in range(0, len(st.session_state.conversation_history), 2):
            if i + 1 < len(st.session_state.conversation_history):
                user_msg = st.session_state.conversation_history[i]
                assistant_msg = st.session_state.conversation_history[i + 1]
                
                with st.container():
                    st.markdown(f"**👤 You:** {user_msg['content']}")
                    st.markdown(f"**🤖 Assistant:** {assistant_msg['content']}")
                    st.divider()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Advanced PDF RAG System | Powered by Azure OpenAI & FAISS</p>
    <p>Supports: Text • Tables • Charts • Graphs • Scanned Documents • Images</p>
</div>
""", unsafe_allow_html=True)