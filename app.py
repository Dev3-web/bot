#!/usr/bin/env python3
"""
Navigation Helper Bot - Streamlit Frontend
Complete frontend interface for managing the navigation helper bot
"""

import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import time
import base64
from io import BytesIO
import sqlite3

# Configure Streamlit page
st.set_page_config(
    page_title="Navigation Helper Bot Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API URL
API_BASE_URL = "http://localhost:8000"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #007bff;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        background: linear-gradient(90deg, #007bff, #0056b3);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .stButton > button:hover {
        background: #0056b3;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'bot_config' not in st.session_state:
    st.session_state.bot_config = {}
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

def make_api_request(endpoint, method="GET", data=None, files=None):
    """Make API requests with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, data=data, timeout=30)
            else:
                headers = {'Content-Type': 'application/json'}
                response = requests.post(url, json=data, headers=headers, timeout=30)
        
        # Check if request was successful
        if response.status_code == 200:
            if response.headers.get('content-type', '').startswith('application/json'):
                return response.json()
            else:
                return response.text
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Make sure the backend is running on http://localhost:8000")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None

def step_progress_bar():
    """Display progress bar for setup steps"""
    steps = ["Upload Sources", "Generate Sitemap", "Set Goals", "Try It Out", "DNS Verification", "Deploy Widget"]
    current = st.session_state.current_step
    
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i + 1 < current:
                st.markdown(f"✅ **{step}**")
            elif i + 1 == current:
                st.markdown(f"🔄 **{step}**")
            else:
                st.markdown(f"⏳ {step}")

def step1_upload_sources():
    """Step 1: Upload sources to knowledge base"""
    st.markdown('<div class="step-header"><h2>📚 Step 1: Upload Sources to Knowledge Base</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>What this does:</strong> Upload documents that your bot will use to answer questions. 
    Supported formats: PDF, DOCX, TXT files.
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload your documents",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt'],
        help="Upload documents that contain information your bot should know about"
    )
    
    if uploaded_files:
        if st.button("Upload Documents", key="upload_docs"):
            with st.spinner("Uploading documents..."):
                # Prepare files for upload
                files_to_upload = []
                for file in uploaded_files:
                    files_to_upload.append(("files", (file.name, file.getvalue(), file.type)))
                
                # Make API request
                try:
                    url = f"{API_BASE_URL}/api/upload-documents"
                    response = requests.post(url, files=files_to_upload)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.uploaded_files = result.get('uploaded_files', [])
                        st.markdown(f"""
                        <div class="success-box">
                        ✅ Successfully uploaded documents!
                        </div>
                        """, unsafe_allow_html=True)
                        time.sleep(1)
                        st.session_state.current_step = 2
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Upload error: {str(e)}")
    
    # Display current knowledge base
    knowledge_base = make_api_request("/api/knowledge-base")
    if knowledge_base and knowledge_base.get('documents'):
        st.subheader("📄 Current Knowledge Base")
        df = pd.DataFrame(knowledge_base['documents'])
        st.dataframe(df, use_container_width=True)
        
        if st.button("Next Step: Generate Sitemap", key="next_step1"):
            st.session_state.current_step = 2
            st.rerun()

def step2_generate_sitemap():
    """Step 2: Generate sitemap automatically"""
    st.markdown('<div class="step-header"><h2>🗺️ Step 2: Generate Website Sitemap</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>What this does:</strong> Automatically crawl your website to understand its structure 
    and help the bot guide users to the right pages.
    </div>
    """, unsafe_allow_html=True)
    
    domain = st.text_input(
        "Enter your website domain",
        placeholder="example.com or https://example.com",
        help="The bot will crawl your website to understand its structure"
    )
    
    if domain and st.button("Generate Sitemap", key="generate_sitemap"):
        with st.spinner("Crawling website and generating sitemap..."):
            result = make_api_request("/api/generate-sitemap", "POST", {"domain": domain})
            
            if result:
                st.markdown(f"""
                <div class="success-box">
                ✅ Sitemap generated successfully! Found {result.get('pages_found', 0)} pages.
                </div>
                """, unsafe_allow_html=True)
                st.session_state.bot_config['domain'] = domain
                time.sleep(1)
                st.session_state.current_step = 3
                st.rerun()
    
    # Display current sitemap
    sitemap = make_api_request("/api/sitemap")
    if sitemap and sitemap.get('sitemap'):
        st.subheader("🌐 Generated Sitemap")
        df = pd.DataFrame(sitemap['sitemap'])
        st.dataframe(df, use_container_width=True)
        
        if st.button("Next Step: Set Target Goals", key="next_step2"):
            st.session_state.current_step = 3
            st.rerun()

def step3_set_goals():
    """Step 3: Define target outcomes and goals"""
    st.markdown('<div class="step-header"><h2>🎯 Step 3: Set Target Outcomes</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>What this does:</strong> Define specific goals your bot should help users achieve. 
    This makes the bot more proactive in guiding users.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Primary Goals")
        goal1 = st.checkbox("Guide users to upload documents", value=True)
        goal2 = st.checkbox("Help users use Q&A feature", value=True)
        goal3 = st.checkbox("Collect customer contact information", value=True)
        goal4 = st.checkbox("Guide users through platform features")
        goal5 = st.checkbox("Provide technical support")
    
    with col2:
        st.subheader("⚙️ Bot Configuration")
        widget_position = st.selectbox(
            "Widget Position",
            ["bottom-right", "bottom-left", "top-right", "top-left"],
            index=0
        )
        
        widget_color = st.color_picker("Widget Color", "#007bff")
        
        custom_goal = st.text_input("Add Custom Goal", placeholder="e.g., Schedule a demo")
    
    # Collect selected goals
    goals = []
    if goal1: goals.append("upload_documents")
    if goal2: goals.append("qa_feature") 
    if goal3: goals.append("collect_contact")
    if goal4: goals.append("platform_guide")
    if goal5: goals.append("technical_support")
    if custom_goal: goals.append(custom_goal)
    
    st.subheader("📝 Selected Goals Preview")
    if goals:
        for i, goal in enumerate(goals, 1):
            st.write(f"{i}. {goal.replace('_', ' ').title()}")
    
    if st.button("Save Configuration", key="save_config"):
        config_data = {
            "domain": st.session_state.bot_config.get('domain', ''),
            "widget_position": widget_position,
            "widget_color": widget_color,
            "goals": goals
        }
        
        result = make_api_request("/api/configure-bot", "POST", config_data)
        
        if result:
            st.session_state.bot_config.update(config_data)
            st.markdown("""
            <div class="success-box">
            ✅ Bot configuration saved successfully!
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1)
            st.session_state.current_step = 4
            st.rerun()

def step4_try_it_out():
    """Step 4: Try it out section"""
    st.markdown('<div class="step-header"><h2>🧪 Step 4: Try It Out</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>What this does:</strong> Test your bot before deploying it to your website. 
    Try different questions and see how it responds.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat with Your Bot")
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = [
                {"role": "bot", "message": "Hello! I'm your navigation helper. How can I assist you today?"}
            ]
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for chat in st.session_state.chat_history:
                if chat["role"] == "user":
                    st.markdown(f"""
                    <div style="padding: 10px; border-radius: 10px; margin: 5px 0; text-align: right;">
                    {chat["message"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="padding: 10px; border-radius: 10px; margin: 5px 0;">
                    {chat["message"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        user_message = st.text_input("Type your message:", key="chat_input")
        
        if st.button("Send Message", key="send_chat") and user_message:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "message": user_message})
            
            # Get bot response
            session_id = "test_session_" + str(int(time.time()))
            chat_data = {"message": user_message, "session_id": session_id}
            result = make_api_request("/api/chat", "POST", chat_data)
            
            if result:
                bot_response = result.get("response", "Sorry, I couldn't process your message.")
                st.session_state.chat_history.append({"role": "bot", "message": bot_response})
            
            st.rerun()
    
    with col2:
        st.subheader("🧪 Test Scenarios")
        
        test_scenarios = [
            "How do I upload documents?",
            "I need help with the Q&A feature",
            "Can you collect my contact information?",
            "What features are available?",
            "I need technical support"
        ]
        
        st.write("**Try these sample questions:**")
        for scenario in test_scenarios:
            if st.button(scenario, key=f"test_{scenario[:10]}"):
                st.session_state.chat_history.append({"role": "user", "message": scenario})
                
                session_id = "test_session_" + str(int(time.time()))
                chat_data = {"message": scenario, "session_id": session_id}
                result = make_api_request("/api/chat", "POST", chat_data)
                
                if result:
                    bot_response = result.get("response", "Sorry, I couldn't process your message.")
                    st.session_state.chat_history.append({"role": "bot", "message": bot_response})
                
                st.rerun()
        
        if st.button("Clear Chat", key="clear_chat"):
            st.session_state.chat_history = [
                {"role": "bot", "message": "Hello! I'm your navigation helper. How can I assist you today?"}
            ]
            st.rerun()
    
    st.markdown("---")
    if st.button("Next Step: DNS Verification", key="next_step4"):
        st.session_state.current_step = 5
        st.rerun()

def step5_dns_verification():
    """Step 5: DNS record verification"""
    st.markdown('<div class="step-header"><h2>🔍 Step 5: DNS Record Verification</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>What this does:</strong> Verify that your domain is accessible and ready for bot deployment. 
    This ensures the widget will work properly on your website.
    </div>
    """, unsafe_allow_html=True)
    
    domain = st.session_state.bot_config.get('domain', '')
    
    if domain:
        st.write(f"**Verifying domain:** {domain}")
        
        if st.button("Verify DNS Records", key="verify_dns"):
            with st.spinner("Verifying DNS records..."):
                result = make_api_request("/api/verify-dns", "POST", {"domain": domain})
                
                if result:
                    if result.get("verified"):
                        st.markdown("""
                        <div class="success-box">
                        ✅ DNS verification successful! Your domain is ready for deployment.
                        </div>
                        """, unsafe_allow_html=True)
                        st.session_state.bot_config['dns_verified'] = True
                        time.sleep(1)
                        st.session_state.current_step = 6
                        st.rerun()
                    else:
                        st.markdown(f"""
                        <div class="warning-box">
                        ⚠️ DNS verification failed: {result.get('message', 'Unknown error')}
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.error("Please complete previous steps first to set your domain.")
        if st.button("Go Back to Step 2", key="back_to_step2"):
            st.session_state.current_step = 2
            st.rerun()
    
    # DNS troubleshooting guide
    with st.expander("🔧 DNS Troubleshooting"):
        st.markdown("""
        **Common DNS Issues:**
        
        1. **Domain not accessible**: Make sure your website is live and accessible
        2. **SSL Certificate**: Ensure your site has a valid SSL certificate (https://)
        3. **Firewall restrictions**: Check if your server blocks external requests
        4. **DNS propagation**: Recent DNS changes may take time to propagate
        
        **Manual verification steps:**
        1. Open your website in a browser
        2. Check if it loads without errors
        3. Verify SSL certificate is valid
        4. Test from different networks/devices
        """)

def step6_deploy_widget():
    """Step 6: Deploy widget and select sections"""
    st.markdown('<div class="step-header"><h2>🚀 Step 6: Deploy Widget</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>What this does:</strong> Generate the widget code and configure where it appears on your website.
    </div>
    """, unsafe_allow_html=True)
    
    domain = st.session_state.bot_config.get('domain', '')
    
    if not domain:
        st.error("Please complete previous steps first.")
        return
    
    # Widget positioning options
    st.subheader("📍 Widget Positioning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        position = st.selectbox(
            "Choose widget position",
            ["bottom-right", "bottom-left", "top-right", "top-left"],
            index=0
        )
        
        pages_to_show = st.multiselect(
            "Select pages where widget should appear",
            ["All pages", "Home page", "Contact page", "About page", "Product pages", "Blog posts"],
            default=["All pages"]
        )
    
    with col2:
        trigger_behavior = st.selectbox(
            "Widget trigger behavior",
            ["Always visible", "Show after 5 seconds", "Show on scroll", "Show on exit intent"]
        )
        
        widget_size = st.selectbox(
            "Widget size",
            ["Compact (300x400)", "Standard (350x500)", "Large (400x600)"]
        )
    
    # Generate widget code
    if st.button("Generate Widget Code", key="generate_widget"):
        with st.spinner("Generating widget code..."):
            result = make_api_request(f"/api/widget-code/{domain}")
            
            if result:
                widget_code = result.get("widget_code", "")
                
                st.subheader("📋 Widget Installation Code")
                
                st.markdown("""
                <div class="success-box">
                ✅ Widget code generated successfully! Copy the code below and paste it into your website.
                </div>
                """, unsafe_allow_html=True)
                
                # Display the widget code
                st.code(widget_code, language="html")
                
                # Download widget code as file
                st.download_button(
                    label="Download Widget Code",
                    data=widget_code,
                    file_name=f"nav-bot-widget-{domain.replace('https://', '').replace('http://', '')}.html",
                    mime="text/html"
                )
                
                # Installation instructions
                with st.expander("📖 Installation Instructions"):
                    st.markdown("""
                    **How to install the widget:**
                    
                    1. **Copy the generated code** above
                    2. **Paste it before the closing `</body>` tag** in your website's HTML
                    3. **For WordPress**: Add to your theme's footer.php file or use a custom HTML widget
                    4. **For other CMS**: Add to your template's footer section
                    5. **Test the widget** by visiting your website
                    
                    **Customization options:**
                    - Modify the `position` value to change widget location
                    - Change the `color` value to match your brand
                    - Adjust the dimensions in the CSS for different sizes
                    """)
                
                # Widget preview
                st.subheader("👀 Widget Preview")
                preview_html = f"""
                <div style="position: relative; height: 100px; background: #f8f9fa; border: 2px dashed #ddd; border-radius: 10px; margin: 20px 0;">
                    <div style="position: absolute; {position.replace('-', ': 20px; ').replace('bottom', 'bottom').replace('top', 'top').replace('left', 'left').replace('right', 'right')}: 20px;">
                        <div style="width: 60px; height: 60px; background: {st.session_state.bot_config.get('widget_color', '#007bff')}; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 24px; cursor: pointer; box-shadow: 0 4px 20px rgba(0,0,0,0.15);">
                            💬
                        </div>
                    </div>
                    <div style="text-align: center; padding-top: 30px; color: #666;">
                        Your website content area<br>
                        <small>Widget will appear in the {position} corner</small>
                    </div>
                </div>
                """
                st.markdown(preview_html, unsafe_allow_html=True)

def dashboard_analytics():
    """Analytics and monitoring dashboard"""
    st.markdown('<div class="step-header"><h2>📊 Analytics Dashboard</h2></div>', unsafe_allow_html=True)
    
    # Sample analytics data (in real implementation, this would come from the database)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Conversations", "247", "12")
    
    with col2:
        st.metric("Documents in KB", len(st.session_state.uploaded_files), "3")
    
    with col3:
        st.metric("Customer Info Collected", "89", "8")
    
    with col4:
        st.metric("Goal Completions", "156", "15")
    
    # Recent conversations
    st.subheader("💬 Recent Conversations")
    
    sample_conversations = [
        {"Time": "2 min ago", "User": "How do I upload?", "Bot": "I can help you upload documents!"},
        {"Time": "5 min ago", "User": "Contact info", "Bot": "I'd be happy to help you get in touch!"},
        {"Time": "8 min ago", "User": "Q&A feature", "Bot": "Great! You can use our Q&A feature..."},
    ]
    
    st.dataframe(pd.DataFrame(sample_conversations), use_container_width=True)
    
    # Customer information collected
    st.subheader("👥 Customer Information")
    
    sample_customers = [
        {"Name": "John Doe", "Email": "john@example.com", "Date": "2024-01-15"},
        {"Name": "Jane Smith", "Email": "jane@example.com", "Date": "2024-01-14"},
        {"Name": "Mike Johnson", "Email": "mike@example.com", "Date": "2024-01-13"},
    ]
    
    st.dataframe(pd.DataFrame(sample_customers), use_container_width=True)

def test_backend_connection():
    """Test if backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Main application function"""
    st.markdown('<h1 class="main-header">🤖 Navigation Helper Bot Dashboard</h1>', unsafe_allow_html=True)
    
    # Test backend connection
    if not test_backend_connection():
        st.error("""
        ❌ **Backend Not Running**
        
        The backend API is not accessible. Please:
        1. Make sure you're running `python main.py` in another terminal
        2. Check if the backend is running on http://localhost:8000
        3. Verify no firewall is blocking the connection
        """)
        st.stop()
    else:
        st.success("✅ Backend connection successful!")
    
    # Sidebar navigation
    st.sidebar.title("🎛️ Navigation")
    
    pages = {
        "🏠 Setup Wizard": "setup",
        "📊 Analytics": "analytics",
        "⚙️ Settings": "settings",
        "❓ Help": "help"
    }
    
    selected_page = st.sidebar.selectbox("Choose a page:", list(pages.keys()))
    
    if pages[selected_page] == "setup":
        # Progress bar
        step_progress_bar()
        
        # Current step display
        if st.session_state.current_step == 1:
            step1_upload_sources()
        elif st.session_state.current_step == 2:
            step2_generate_sitemap()
        elif st.session_state.current_step == 3:
            step3_set_goals()
        elif st.session_state.current_step == 4:
            step4_try_it_out()
        elif st.session_state.current_step == 5:
            step5_dns_verification()
        elif st.session_state.current_step == 6:
            step6_deploy_widget()
        
        # Step navigation in sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Quick Navigation")
        
        for i in range(1, 7):
            step_names = ["Upload Sources", "Generate Sitemap", "Set Goals", "Try It Out", "DNS Verification", "Deploy Widget"]
            if st.sidebar.button(f"Step {i}: {step_names[i-1]}", key=f"nav_step_{i}"):
                st.session_state.current_step = i
                st.rerun()
    
    elif pages[selected_page] == "analytics":
        dashboard_analytics()
    
    elif pages[selected_page] == "settings":
        st.markdown('<div class="step-header"><h2>⚙️ Bot Settings</h2></div>', unsafe_allow_html=True)
        
        st.subheader("🤖 Current Configuration")
        if st.session_state.bot_config:
            for key, value in st.session_state.bot_config.items():
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        else:
            st.info("No configuration found. Please complete the setup wizard first.")
        
        st.subheader("🔄 Reset Configuration")
        if st.button("Reset All Settings", type="secondary"):
            st.session_state.clear()
            st.success("Configuration reset successfully!")
            st.rerun()
    
    elif pages[selected_page] == "help":
        st.markdown('<div class="step-header"><h2>❓ Help & Documentation</h2></div>', unsafe_allow_html=True)
        
        with st.expander("🚀 Getting Started"):
            st.markdown("""
            1. **Upload Sources**: Add documents your bot should know about
            2. **Generate Sitemap**: Let the bot learn your website structure
            3. **Set Goals**: Define what you want the bot to help users achieve
            4. **Try It Out**: Test your bot before deployment
            5. **DNS Verification**: Ensure your domain is ready
            6. **Deploy Widget**: Get the code and install on your website
            """)
        
        with st.expander("🔧 Technical Requirements"):
            st.markdown("""
            **Backend Requirements:**
            - Python 3.8+
            - FastAPI
            - SQLite
            - Required packages: requests, beautifulsoup4, PyPDF2, python-docx
            
            **Frontend Requirements:**
            - Streamlit
            - pandas
            - requests
            
            **Installation:**
            ```bash
            pip install fastapi uvicorn streamlit pandas requests beautifulsoup4 PyPDF2 python-docx
            ```
            """)
        
        with st.expander("❓ Troubleshooting"):
            st.markdown("""
            **Common Issues:**
            
            1. **API Connection Error**: Make sure the backend is running on port 8000
            2. **File Upload Fails**: Check file formats (PDF, DOCX, TXT only)
            3. **Sitemap Generation Fails**: Ensure website is accessible
            4. **Widget Not Showing**: Check if code is placed before `</body>` tag
            
            **Support:**
            - Check the console for error messages
            - Ensure all required packages are installed
            - Verify API endpoints are accessible
            """)

if __name__ == "__main__":
    main()