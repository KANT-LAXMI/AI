import streamlit as st
import pandas as pd
from datetime import datetime
import time
from agent import PDLAgent
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="PDL Professional Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        color: white;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #f0f0f0;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 25px;
        padding: 15px 40px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #667eea;
        padding: 12px;
        font-size: 1rem;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* Stats card */
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .stat-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'agent' not in st.session_state:
    st.session_state.agent = PDLAgent()
if 'current_results' not in st.session_state:
    st.session_state.current_results = None

# Header
st.markdown('<h1 class="main-title">🔍 Professional Data Search</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Discover professionals across industries with AI-powered search</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image(r"C:\Users\LaxmiKant\OneDrive - C3IT Software Solutions Pvt. Ltd\OLD_DATA\Desktop\self\linkedin_agent\New folder\agent_logo.jpg",use_container_width=True)
    st.title("Search Dashboard")
    st.markdown("---")
    
    # Search history
    st.subheader("📊 Recent Searches")
    if st.session_state.search_history:
        for item in st.session_state.search_history[-5:]:
            with st.expander(f"🔹 {item['keyword']}", expanded=False):
                st.write(f"**Total Found:** {item['total']}")
                st.write(f"**Date:** {item['timestamp']}")
    else:
        st.info("No searches yet. Start exploring!")
    
    st.markdown("---")
    
    # Statistics
    if st.session_state.search_history:
        total_searches = len(st.session_state.search_history)
        total_profiles = sum(item['total'] for item in st.session_state.search_history)
        
        st.metric("Total Searches", total_searches, delta=None)
        st.metric("Total Profiles Found", total_profiles, delta=None)

# Main content area
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
 
    keyword = st.text_input(
        "Enter your search keyword",
        placeholder="e.g., Software Engineer, Marketing Manager, Data Scientist...",
        help="Enter a job title, skill, or keyword to search for professionals",
        label_visibility="collapsed"
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        search_button = st.button("🚀 Start Search", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Search execution
if search_button and keyword:
    with st.spinner(''):
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            st.markdown("<div style='background: white; padding: 20px; border-radius: 15px; margin-top: 20px;'>", unsafe_allow_html=True)
            
            # Progress bar with steps
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Initialize
            status_text.markdown("**🔄 Initializing search...**")
            progress_bar.progress(20)
            time.sleep(0.5)
            
            # Step 2: Searching
            status_text.markdown("**🔍 Searching**")
            progress_bar.progress(50)
            
            try:
                result = st.session_state.agent.run(keyword)
                
                # Step 3: Processing
                status_text.markdown("**⚙️ Processing results...**")
                progress_bar.progress(75)
                time.sleep(0.5)
                
                # Step 4: Exporting
                status_text.markdown("**📊 Generating Excel file...**")
                progress_bar.progress(90)
                time.sleep(0.5)
                
                # Step 5: Complete
                status_text.markdown("**✅ Search completed successfully!**")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                # Store results
                st.session_state.current_results = result
                st.session_state.search_history.append({
                    'keyword': keyword,
                    'total': result['total'],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Success animation
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.markdown("</div>", unsafe_allow_html=True)

# Results display
if st.session_state.current_results:
    result = st.session_state.current_results
    
    st.markdown("---")
    st.markdown("## 📈 Search Results")
    
    # Stats cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='stat-box'>
            <div class='stat-label'>Search Keyword</div>
            <div class='stat-number' style='font-size: 1.8rem;'>{result['keyword']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='stat-box'>
            <div class='stat-label'>Total Professionals</div>
            <div class='stat-number'>{result['total']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='stat-box'>
            <div class='stat-label'>Records Exported</div>
            <div class='stat-number'>25</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Download button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        try:
            with open(result['file'], 'rb') as file:
                st.download_button(
                    label="📥 Download Excel Report",
                    data=file,
                    file_name=result['file'],
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        except:
            st.warning("Excel file not found. Please run the search again.")
    
    # Display sample data
    st.markdown("---")
    st.markdown("### 👥 Sample Profiles Preview")
    
    try:
        df = pd.read_excel(result['file'], sheet_name='Profiles')
        
        # Display in a nice table
        st.dataframe(
            df.head(10),
            use_container_width=True,
            height=400
        )
        
        # Visualizations
        if len(df) > 0:
            st.markdown("---")
            st.markdown("### 📊 Visual Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Matched fields distribution
                if 'Matched Fields' in df.columns:
                    matched_counts = df['Matched Fields'].str.split(', ').explode().value_counts()
                    
                    fig1 = px.bar(
                        x=matched_counts.index,
                        y=matched_counts.values,
                        title="Match Distribution by Field",
                        labels={'x': 'Field', 'y': 'Count'},
                        color=matched_counts.values,
                        color_continuous_scale='Purples'
                    )
                    fig1.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Top companies
                if 'Company' in df.columns:
                    top_companies = df['Company'].value_counts().head(10)
                    
                    fig2 = px.pie(
                        values=top_companies.values,
                        names=top_companies.index,
                        title="Top 10 Companies",
                        color_discrete_sequence=px.colors.sequential.Purples
                    )
                    fig2.update_layout(height=400)
                    st.plotly_chart(fig2, use_container_width=True)
            
    except Exception as e:
        st.warning(f"Could not load preview: {str(e)}")

# Footer
st.markdown("---")
# st.markdown("""
#     <div style='text-align: center; color: white; padding: 20px;'>
#         <p style='font-size: 0.9rem; opacity: 0.8;'>
#             Powered by People Data Labs API | Built with ❤️ using Streamlit
#         </p>
#     </div>
# """, unsafe_allow_html=True)