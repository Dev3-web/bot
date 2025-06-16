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
import re
from typing import List, Dict, Set
import sqlite3

# Configure Streamlit page
st.set_page_config(
    page_title="Navigation Helper Bot Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API URL
API_BASE_URL = "http://127.0.0.1:8000"

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
    .info-box {
        background: #e7f3ff;
        border: 1px solid #cce5ff;
        color: #004085;
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
if 'suggested_goals' not in st.session_state: # Ensure this is initialized
    st.session_state.suggested_goals = []
# NEW: Flag to indicate if we just transitioned to step 3 from step 2
if '_just_entered_step3_from_step2' not in st.session_state:
    st.session_state._just_entered_step3_from_step2 = False

def make_api_request(endpoint, method="GET", data=None, files=None):
    """Make API requests with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"

        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, data=data)
            else:
                headers = {'Content-Type': 'application/json'}
                response = requests.post(url, json=data, headers=headers)

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
                        st.session_state.uploaded_files = result.get('results', []) # Changed key to 'results' based on backend
                        st.markdown(f"""
                        <div class="success-box">
                        ✅ Successfully uploaded documents! Processed {result.get('successful_rag_additions', 0)} files.
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
        # Convert upload_date to a readable format
        df['upload_date'] = pd.to_datetime(df['upload_date']).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(df, use_container_width=True)

    # Always show the next step button if not currently uploading
    if st.session_state.current_step == 1: # Only show if in the current step
        if st.button("Next Step: Generate Sitemap", key="next_step1_bottom"):
            st.session_state.current_step = 2
            st.rerun()

def step2_generate_sitemap():
    """Step 2: Generate sitemap automatically"""
    st.markdown('<div class="step-header"><h2>🗺️ Step 2: Generate Website Sitemap & Goals</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>What this does:</strong> Enter your website's domain. The bot will crawl your site to build a sitemap and
    suggest common user goals based on its content. This data enhances the bot's navigation capabilities.
    </div>
    """, unsafe_allow_html=True)

    domain = st.text_input(
        "Enter your website domain",
        placeholder="example.com or https://example.com",
        help="The bot will crawl your website to understand its structure and suggest goals."
    )

    if domain and st.button("Generate Sitemap and Suggest Goals", key="generate_sitemap"):
        with st.spinner("Initiating crawl..."):
            status_placeholder = st.empty()
            crawl_messages = [
                "🌐 Crawling homepage...",
                "🔍 Discovering new links...",
                "📄 Extracting page content...",
                "🚀 Following internal links...",
                "📊 Analyzing content structure...",
                "✨ Using AI to suggest user goals...",
                "✅ Almost done with indexing..."
            ]

            # Simulate crawling activity before the actual API call returns
            for i, msg in enumerate(crawl_messages):
                status_placeholder.info(f"{msg} ({i+1}/{len(crawl_messages)})")
                time.sleep(1.5) # Adjust sleep time for desired speed

            # After simulation, make the actual API call
            status_placeholder.info("Sending crawl request to backend and waiting for full results...")
            result = make_api_request("/api/generate-sitemap", "POST", {"domain": domain})

            if result:
                status_placeholder.success("Crawl completed and goals suggested!")
                st.markdown(f"""
                <div class="success-box">
                ✅ Sitemap generated and goals suggested successfully!
                <br><strong>Pages Discovered:</strong> {result.get('metrics', {}).get('pages_discovered', 'N/A')}
                <br><strong>RAG Chunks Added:</strong> {result.get('metrics', {}).get('rag_chunks_added', 'N/A')}
                </div>
                """, unsafe_allow_html=True)

                st.session_state.bot_config['domain'] = domain
                st.session_state['suggested_goals'] = result.get('suggested_goals', [])

                # NEW: Set flag to indicate fresh entry into step 3
                st.session_state._just_entered_step3_from_step2 = True

                time.sleep(1)
                st.session_state.current_step = 3
                st.rerun()
            else:
                status_placeholder.error("Crawl failed. Please check the domain and backend logs.")


    # Display current sitemap
    sitemap = make_api_request("/api/sitemap")
    if sitemap and sitemap.get('sitemap'):
        st.subheader("🌐 Generated Sitemap (Partial View)")
        df = pd.DataFrame(sitemap['sitemap'])
        # Limit columns for display to avoid clutter
        display_cols = ['url', 'title', 'depth', 'description', 'og_title', 'publication_date']
        # Filter out columns that don't exist in the current data
        existing_cols = [col for col in display_cols if col in df.columns]
        # Convert publication_date to a readable format if it exists
        if 'publication_date' in existing_cols:
            df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(df[existing_cols], use_container_width=True)

    if st.button("Next Step: Set Target Goals", key="next_step2_bottom"):
        st.session_state.current_step = 3
        st.rerun()


def step3_set_goals():
    """Step 3: Define target outcomes and goals (with dynamic multi-goal support)"""
    st.markdown('<div class="step-header"><h2>🎯 Step 3: Set Target Outcomes</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>What this does:</strong> Select suggested goals or add multiple custom goals that your bot should help users achieve.
    These goals guide the bot's conversational flow and its ability to direct users.
    </div>
    """, unsafe_allow_html=True)

    # NEW: Conditional reset of goals when entering this step from sitemap generation
    if st.session_state._just_entered_step3_from_step2:
        st.session_state.goal_states = {goal: True for goal in st.session_state.get('suggested_goals', [])}
        st.session_state.custom_goals = []
        st.session_state._just_entered_step3_from_step2 = False # Reset the flag
        st.rerun() # Rerun to apply the fresh state immediately

    # --- STATE INITIALIZATION (now simplified, as reset happens above) ---
    # Only initialize if these don't exist AT ALL (first app run or full clear)
    if 'goal_states' not in st.session_state:
        st.session_state.goal_states = {goal: True for goal in st.session_state.get('suggested_goals', [])}

    if 'custom_goals' not in st.session_state:
        st.session_state.custom_goals = []

    # --- CALLBACK FUNCTION TO ADD A CUSTOM GOAL ---
    def add_custom_goal():
        """Adds the content of the text_input to the custom_goals list."""
        goal_text = st.session_state.get("custom_goal_input", "").strip()
        if goal_text and goal_text not in st.session_state.custom_goals:
            st.session_state.custom_goals.append(goal_text)
            # Clear the input box after adding
            st.session_state.custom_goal_input = ""

    # --- UI LAYOUT ---
    col1, col2 = st.columns(2)

    with col1:
        # --- SUGGESTED GOALS ---
        st.subheader("🎯 Suggested Goals")
        suggested_goals = st.session_state.get('suggested_goals', [])
        if suggested_goals:
            for goal in suggested_goals:
                # Ensure the goal exists in goal_states before setting checkbox state
                if goal not in st.session_state.goal_states:
                    st.session_state.goal_states[goal] = True # Default to true for new suggested goals
                st.session_state.goal_states[goal] = st.checkbox(
                    goal,
                    value=st.session_state.goal_states.get(goal, True),
                    key=f"goal_{goal}"
                )
        else:
            st.info("No specific goals were automatically identified from the website. Please add custom goals.")

        # --- CUSTOM GOALS INPUT ---
        st.subheader("➕ Add Custom Goals")
        st.text_input(
            "Enter a new goal and press Enter",
            key="custom_goal_input",
            on_change=add_custom_goal,
            placeholder="e.g., Book a consultation"
        )
        st.button("Add Goal", on_click=add_custom_goal)

        # --- DISPLAY CURRENT CUSTOM GOALS WITH DELETE OPTION ---
        if st.session_state.custom_goals:
            st.write("Your custom goals:")
            # We iterate in reverse to safely delete items without index errors
            for i in range(len(st.session_state.custom_goals) - 1, -1, -1):
                goal_item = st.session_state.custom_goals[i]
                row_col1, row_col2 = st.columns([0.85, 0.15])
                with row_col1:
                    st.write(f"• {goal_item}")
                with row_col2:
                    if st.button("🗑️", key=f"delete_custom_{i}", help=f"Delete goal: {goal_item}"):
                        st.session_state.custom_goals.pop(i)
                        st.rerun() # Force an immediate rerun to update the list display

    with col2:
        # --- BOT CONFIGURATION ---
        st.subheader("⚙️ Bot Configuration")
        widget_position = st.selectbox(
            "Widget Position",
            ["bottom-right", "bottom-left", "top-right", "top-left"],
            index=0
        )
        widget_color = st.color_picker("Widget Color", "#007bff")

    # --- FINAL GOAL COMPILATION AND PREVIEW ---
    st.divider()
    st.subheader("📝 Selected Goals Preview")

    # Collect all selected goals
    final_goals = []
    # 1. Add selected suggested goals
    for goal, is_selected in st.session_state.get('goal_states', {}).items():
        if is_selected:
            final_goals.append(goal)
    # 2. Add all custom goals (ensure no duplicates with suggested ones)
    for custom_goal in st.session_state.custom_goals:
        if custom_goal not in final_goals:
            final_goals.append(custom_goal)


    if final_goals:
        for i, goal in enumerate(final_goals, 1):
            st.markdown(f"`{i}.` {goal}")
    else:
        st.warning("No goals have been selected or added.")

    # --- SAVE BUTTON ---
    if st.button("Save Configuration & Continue", key="save_config"):
        if not final_goals:
            st.error("Please select or add at least one goal before saving.")
        else:
            config_data = {
                "domain": st.session_state.bot_config.get('domain', ''),
                "widget_position": widget_position,
                "widget_color": widget_color,
                "goals": final_goals # Use the dynamically compiled list
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
    Try different questions and see how it responds. The bot leverages the documents and sitemap you provided.
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
        chat_container = st.container(height=300, border=True) # Fixed height for chat container
        with chat_container:
            for chat in st.session_state.chat_history:
                if chat["role"] == "user":
                    st.markdown(f"""
                    <div style="background-color: #007bff; color: white; padding: 10px; border-radius: 10px; margin: 5px 0 5px auto; max-width: 80%; float: right;">
                    {chat["message"]}
                    </div>
                    <div style="clear: both;"></div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; color: #333; padding: 10px; border-radius: 10px; margin: 5px auto 5px 0; max-width: 80%; float: left;">
                    {chat["message"]}
                    </div>
                    <div style="clear: both;"></div>
                    """, unsafe_allow_html=True)

        # Chat input
        user_message = st.text_input("Type your message:", key="chat_input", on_change=lambda: st.session_state.get("chat_input", "").strip() and st.button("Send Message", key="send_chat"))


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

            # Clear input after sending and rerun to refresh chat
            st.session_state.chat_input = ""
            st.rerun()

    with col2:
        st.subheader("🧪 Test Scenarios")

        # Use the configured goals from session_state if available, otherwise default
        configured_goals = st.session_state.bot_config.get('goals', [])
        if not configured_goals:
             configured_goals = st.session_state.get('suggested_goals', []) # Fallback to suggested if no config saved

        test_scenarios = configured_goals[:5] # Take up to 5 of the configured/suggested goals
        if not test_scenarios:
            test_scenarios = [
                "How do I upload documents?",
                "I need help with the Q&A feature",
                "Can you collect my contact information?",
                "What features are available?",
                "I need technical support"
            ]
            st.info("No configured goals found. Showing general test scenarios.")
        else:
            st.info("Test your bot with the goals you've configured!")


        st.write("**Try these sample questions:**")
        for scenario in test_scenarios:
            if st.button(scenario, key=f"test_{scenario[:20].replace(' ', '_').lower()}"):
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
    This ensures the widget will work properly on your website by checking basic connectivity.
    </div>
    """, unsafe_allow_html=True)

    domain = st.session_state.bot_config.get('domain', '')

    if domain:
        st.write(f"**Verifying domain:** `{domain}`")

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
        st.error("Please complete Step 2 to set your domain before verifying DNS.")
        if st.button("Go Back to Step 2", key="back_to_step2"):
            st.session_state.current_step = 2
            st.rerun()

    # DNS troubleshooting guide
    with st.expander("🔧 DNS Troubleshooting"):
        st.markdown("""
        **Common DNS Issues:**

        1.  **Domain not accessible**: Make sure your website is live and accessible from the public internet (e.g., not behind a local firewall only).
        2.  **SSL Certificate**: Ensure your site has a valid SSL certificate (HTTPS) and that it's properly configured. The verification might fail for sites with invalid or expired certificates.
        3.  **Firewall restrictions**: Check if your server or hosting provider blocks external requests from services like this bot.
        4.  **DNS propagation**: If you've just made DNS changes, it might take some time (up to 48 hours) for them to propagate globally.

        **Manual verification steps:**
        1.  Open your website in a browser using `https://yourdomain.com`
        2.  Check if it loads without errors and if the padlock icon (SSL certificate) is valid.
        3.  Try accessing your site from different internet connections or devices to rule out local network issues.
        """)
    if st.session_state.bot_config.get('dns_verified', False):
        if st.button("Next Step: Deploy Widget", key="next_step5_bottom"):
            st.session_state.current_step = 6
            st.rerun()


def step6_deploy_widget():
    """Step 6: Deploy widget and select sections"""
    st.markdown('<div class="step-header"><h2>🚀 Step 6: Deploy Widget</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>What this does:</strong> Generate the HTML snippet for your bot widget. You'll embed this code into your website's
    HTML to make the bot appear and function. Basic positioning and size options are available.
    </div>
    """, unsafe_allow_html=True)

    domain = st.session_state.bot_config.get('domain', '')
    widget_color = st.session_state.bot_config.get('widget_color', '#007bff')
    widget_position = st.session_state.bot_config.get('widget_position', 'bottom-right')

    if not domain:
        st.error("Please complete Step 2 to set your domain before deploying the widget.")
        return

    # Fetch sitemap to populate page selection (for potential future enhancements like page-specific deployment)
    # For now, this is just to show that the data is available
    sitemap_data = make_api_request("/api/sitemap")

    st.subheader("📍 Widget Positioning")

    col1, col2 = st.columns(2)

    with col1:
        # These are now derived from bot_config but can be made editable if we want to save changes here
        st.write(f"**Current Widget Position:** `{widget_position.replace('-', ' ')}`")
        st.write(f"**Current Widget Color:** `{widget_color}`")

        # Example of making them editable again, and requiring re-saving config if changed
        # new_position = st.selectbox(
        #     "Choose widget position",
        #     ["bottom-right", "bottom-left", "top-right", "top-left"],
        #     index=["bottom-right", "bottom-left", "top-right", "top-left"].index(widget_position)
        # )
        # new_color = st.color_picker("Widget Color", widget_color)

    with col2:
        # These are purely for illustrative purposes in the UI, not passed to backend yet
        st.info("Further customization (like size, trigger behavior) can be done by manually editing the generated code.")
        st.selectbox(
            "Widget size (Informational)",
            ["Standard (380x600)", "Compact (300x500)", "Large (450x700)"] # Default is 380x600 in widget.html
        )
        st.selectbox(
            "Widget trigger behavior (Informational)",
            ["Click to open", "Show after 5 seconds", "Show on scroll", "Show on exit intent"]
        )


    # Generate widget code
    if st.button("Generate Widget Code", key="generate_widget"):
        with st.spinner("Generating widget code..."):
            # We pass the domain (which is stored in bot_config) to get the specific config
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
                    file_name=f"nav-bot-widget-{domain.replace('https://', '').replace('http://', '').replace('/', '_')}.html",
                    mime="text/html"
                )

                # Installation instructions
                with st.expander("📖 Installation Instructions"):
                    st.markdown("""
                    **How to install the widget:**

                    1.  **Copy the generated code** above.
                    2.  **Paste it before the closing `</body>` tag** in your website's HTML file. This is usually in your theme's `footer.php` (for WordPress), a global footer template, or directly in the HTML of pages where you want the bot to appear.
                    3.  **For WordPress users**: You can often use a "Custom HTML" widget in your footer or a plugin that allows injecting code into the footer.
                    4.  **For other CMS/Website Builders**: Look for options to embed custom HTML/JavaScript, usually in a "footer scripts" or "header/footer code" section.
                    5.  **Test the widget** by visiting your website in a browser. Clear your browser cache if it doesn't appear immediately.

                    **Customization options (advanced - edit the code directly):**
                    -   Modify the `position` value in the JavaScript part of the widget code to change its exact corner (e.g., `bottom-right`, `bottom-left`).
                    -   Change the `color` value (e.g., `#007bff`) to match your brand's primary color.
                    -   Adjust the `width` and `height` properties in the `botContainer.style.cssText` for different widget sizes.
                    """)

                # Widget preview (simplified, as actual iframe would need cross-origin setup)
                st.subheader("👀 Widget Preview")
                preview_html = f"""
                <div style="position: relative; height: 100px; background: #f8f9fa; border: 2px dashed #ddd; border-radius: 10px; margin: 20px 0; overflow: hidden;">
                    <div style="position: absolute; {widget_position.replace('-', ': 20px; ').replace('bottom', 'bottom').replace('top', 'top').replace('left', 'left').replace('right', 'right')}: 20px;">
                        <div style="width: 60px; height: 60px; background: {widget_color}; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 24px; cursor: pointer; box-shadow: 0 4px 20px rgba(0,0,0,0.15);">
                            💬
                        </div>
                    </div>
                    <div style="text-align: center; padding-top: 30px; color: #666;">
                        Your website content area<br>
                        <small>Widget will appear in the {widget_position.replace('-', ' ')} corner</small>
                    </div>
                </div>
                """
                st.markdown(preview_html, unsafe_allow_html=True)
            else:
                st.error("Failed to generate widget code. Ensure your domain is correct and backend is running.")


def test_backend_connection():
    """Test if backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        return response.status_code == 200
    except:
        return False

def main():
    """Main application function"""
    st.markdown('<h1 class="main-header">🤖 Navigation Helper Bot Dashboard</h1>', unsafe_allow_html=True)

    # Test backend connection (uncomment to enable)
    # if not test_backend_connection():
    #     st.error("""
    #     ❌ **Backend Not Running**
    #
    #     The backend API is not accessible. Please:
    #     1. Make sure you're running `python main.py` in another terminal
    #     2. Check if the backend is running on http://localhost:8000
    #     3. Verify no firewall is blocking the connection
    #     """)
    #     st.stop()
    # else:
    #     st.success("✅ Backend connection successful!")

    # Sidebar navigation
    st.sidebar.title("🎛️ Navigation")

    pages = {
        "🏠 Setup Wizard": "setup",
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

    elif pages[selected_page] == "settings":
        st.markdown('<div class="step-header"><h2>⚙️ Bot Settings</h2></div>', unsafe_allow_html=True)

        st.subheader("🤖 Current Configuration")
        if st.session_state.bot_config:
            # Fetch the most recent config from DB to display
            domain = st.session_state.bot_config.get('domain')
            if domain:
                try:
                    # Use the context manager for database connection
                    with sqlite3.connect("nav_bot.db") as conn:
                        conn.row_factory = sqlite3.Row
                        cursor = conn.cursor()
                        cursor.execute("SELECT * FROM bot_config WHERE domain = ?", (domain,))
                        db_config = cursor.fetchone()
                        if db_config:
                            # Update session state with DB values
                            st.session_state.bot_config['widget_position'] = db_config['widget_position']
                            st.session_state.bot_config['widget_color'] = db_config['widget_color']
                            st.session_state.bot_config['goals'] = json.loads(db_config['goals']) if db_config['goals'] else []
                            st.session_state.bot_config['dns_verified'] = bool(db_config['dns_verified']) # Ensure boolean
                            st.session_state.bot_config['created_date'] = db_config['created_date']

                            for key, value in st.session_state.bot_config.items():
                                if key == 'goals' and isinstance(value, list):
                                    st.write(f"**{key.replace('_', ' ').title()}:**")
                                    for i, goal in enumerate(value):
                                        st.write(f"  - {goal}")
                                else:
                                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                        else:
                            st.warning(f"No configuration found in database for domain: {domain}. Displaying session state.")
                            for key, value in st.session_state.bot_config.items():
                                if key == 'goals' and isinstance(value, list):
                                    st.write(f"**{key.replace('_', ' ').title()}:**")
                                    for i, goal in enumerate(value):
                                        st.write(f"  - {goal}")
                                else:
                                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                except Exception as e:
                    st.error(f"Error loading configuration from database: {e}")
                    st.info("Displaying current session state configuration.")
                    for key, value in st.session_state.bot_config.items():
                        if key == 'goals' and isinstance(value, list):
                            st.write(f"**{key.replace('_', ' ').title()}:**")
                            for i, goal in enumerate(value):
                                st.write(f"  - {goal}")
                        else:
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            else:
                 st.info("No domain configured yet. Please complete the setup wizard.")
        else:
            st.info("No configuration found. Please complete the setup wizard first.")

        st.subheader("🔄 Reset Configuration and Data")
        st.warning("This will clear all uploaded documents, sitemap data, and bot configurations from the database and reset the dashboard.")
        if st.button("Clear All Data", type="secondary"):
            if make_api_request("/api/knowledge-base", method="DELETE"):
                st.session_state.clear() # Clears ALL session state variables
                st.success("All data cleared and configuration reset successfully! Please refresh the page if needed.")
                st.rerun()

    elif pages[selected_page] == "help":
        st.markdown('<div class="step-header"><h2>❓ Help & Documentation</h2></div>', unsafe_allow_html=True)

        with st.expander("🚀 Getting Started"):
            st.markdown("""
            This dashboard guides you through setting up your AI-powered navigation helper bot.

            **Here's the typical workflow:**
            1.  **Upload Sources**: Provide documents (PDF, DOCX, TXT) with information your bot should use to answer user questions. This forms the bot's "knowledge base."
            2.  **Generate Sitemap**: Enter your website's domain. The bot will crawl your site to understand its structure, extract content, and suggest user goals based on the content it finds. This data powers the bot's ability to provide navigation assistance.
            3.  **Set Goals**: Review the AI-suggested goals and add your own custom goals. These are the key actions or information users will likely seek on your site, guiding the bot's responses.
            4.  **Try It Out**: Interact with your bot in a test environment to see how it responds to different queries. This is a crucial step to fine-tune its performance.
            5.  **DNS Verification**: A quick check to ensure your domain is publicly accessible and ready for bot deployment. This helps prevent issues when you embed the widget.
            6.  **Deploy Widget**: Generate the HTML code snippet for your bot widget. You'll then embed this code into your website's `</body>` section to make the bot live.
            """)

        with st.expander("🔧 Technical Requirements"):
            st.markdown("""
            **Backend (FastAPI) Requirements:**
            -   Python 3.8+
            -   `fastapi`, `uvicorn`, `requests`, `beautifulsoup4`, `PyPDF2`, `python-docx`, `python-dotenv`, `langchain-community`, `langchain-openai`, `chromadb`, `aiohttp`
            -   An `OPENAI_API_KEY` set as an environment variable or in a `.env` file for AI features.
            -   Runs on `http://localhost:8000` by default.

            **Frontend (Streamlit) Requirements:**
            -   Python 3.8+
            -   `streamlit`, `pandas`, `requests`
            -   Runs the Streamlit app.

            **Installation Steps:**
            1.  **Clone the repository** (if applicable) or ensure all Python files are in the same directory.
            2.  **Install dependencies**:
                ```bash
                pip install fastapi uvicorn streamlit pandas requests beautifulsoup4 PyPDF2 python-docx python-dotenv langchain-community langchain-openai chromadb aiohttp
                ```
            3.  **Set your OpenAI API Key**: Create a `.env` file in the same directory as your `main.py` (backend) and add:
                ```
                OPENAI_API_KEY="your_openai_api_key_here"
                ```
            4.  **Run the Backend**: Open your terminal/command prompt, navigate to the project directory, and run:
                ```bash
                uvicorn main:app --reload --port 8000
                ```
            5.  **Run the Frontend**: Open *another* terminal/command prompt, navigate to the project directory, and run:
                ```bash
                streamlit run frontend_app.py
                ```
            Your browser should automatically open the Streamlit dashboard.
            """)

        with st.expander("❓ Troubleshooting"):
            st.markdown("""
            **Common Issues & Solutions:**

            1.  **API Connection Error (Backend Not Running)**:
                -   **Solution**: Ensure you have successfully started the backend API in a separate terminal using `uvicorn main:app --reload --port 8000`. Verify there are no errors in the backend terminal output.

            2.  **`OPENAI_API_KEY` not set**:
                -   **Solution**: The backend `main.py` requires `OPENAI_API_KEY` to be set. Create a `.env` file in the same directory as `main.py` and add `OPENAI_API_KEY="your_key_here"`. Restart the backend.

            3.  **File Upload Fails**:
                -   **Solution**: Check the file formats. Only PDF, DOCX, and TXT are supported. Ensure the files are not corrupted. Check backend logs for specific errors during processing.

            4.  **Sitemap Generation Fails or Finds Zero Pages**:
                -   **Solution**:
                    -   Verify the domain is correct and accessible from the public internet (try opening it in your browser).
                    -   Ensure the website is not blocking crawlers (e.g., via `robots.txt` or aggressive WAF/firewall rules).
                    -   Check backend logs for HTTP status codes or specific errors during the crawl.

            5.  **Widget Not Showing on Website**:
                -   **Solution**:
                    -   Ensure the generated widget code is pasted *before* the closing `</body>` tag of your website's HTML.
                    -   Clear your browser's cache and cookies for your website.
                    -   Inspect your website's console (F12 in most browsers) for JavaScript errors.
                    -   Verify that `http://localhost:8000` is accessible from the client's browser where the widget is embedded (this might require deploying the backend to a public server if embedding on a live website).

            6.  **`sqlite3.OperationalError: no such column`**:
                -   **Solution**: This typically means you've updated the backend schema (`init_database` function) but your `nav_bot.db` file is old.
                -   **Option A (Recommended for development)**: Delete the `nav_bot.db` file and restart the backend. It will re-create the database with the new schema.
                -   **Option B**: The `init_database` function now includes `ALTER TABLE` statements to add new columns safely, so restarting the backend *should* update it. If not, delete `nav_bot.db`.

            For further support, check the console output of both your Streamlit frontend and FastAPI backend for detailed error messages.
            """)

if __name__ == "__main__":
    main()