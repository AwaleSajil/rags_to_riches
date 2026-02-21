import streamlit as st
import asyncio
import os
import json
import plotly.io as pio
from supabase import create_client, Client, ClientOptions
from dotenv import load_dotenv

from money_rag import MoneyRAG

load_dotenv()

st.set_page_config(page_title="MoneyRAG", layout="wide", initial_sidebar_state="expanded")

# Initialize Supabase Client per request (NO CACHE) to ensure thread-safe auth headers
def get_supabase() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if "access_token" in st.session_state:
        opts = ClientOptions(headers={"Authorization": f"Bearer {st.session_state.access_token}"})
        return create_client(url, key, options=opts)
    return create_client(url, key)

supabase = get_supabase()

def inject_css():
    st.html("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
    /* ── Global ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem !important; }

    /* ── Background ── */
    .stApp { background: #f8f9ff; color: #1e1e3a; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: white !important;
        border-right: 1px solid #e8eaf6 !important;
        box-shadow: 2px 0 12px rgba(99,102,241,0.06) !important;
    }
    [data-testid="stSidebar"] * { color: #374151 !important; }

    /* ── Sidebar Nav Buttons ── */
    div[data-testid="stSidebarContent"] .nav-btn > div > button {
        width: 100% !important; text-align: left !important;
        border: none !important; border-radius: 10px !important;
        background: transparent !important; color: #6b7280 !important;
        padding: 0.65rem 1rem !important; font-size: 0.9rem !important;
        font-weight: 500 !important; transition: all 0.18s ease !important;
    }
    div[data-testid="stSidebarContent"] .nav-btn > div > button:hover {
        background: #f0f1ff !important; color: #6366f1 !important;
    }
    div[data-testid="stSidebarContent"] .nav-btn-active > div > button {
        background: linear-gradient(135deg, #eef2ff, #f5f3ff) !important;
        color: #6366f1 !important;
        border: 1px solid #c7d2fe !important;
        font-weight: 600 !important;
    }

    /* ── Primary Buttons ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border: none !important; border-radius: 10px !important;
        color: white !important; font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
    }

    /* ── Secondary Buttons ── */
    .stButton > button[kind="secondary"] {
        background: white !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 10px !important; color: #374151 !important;
        font-weight: 500 !important; transition: all 0.18s ease !important;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #6366f1 !important; color: #6366f1 !important;
        background: #f0f1ff !important;
    }

    /* ── Inputs ── */
    .stTextInput input,
    .stTextInput input[type="password"],
    .stTextArea textarea,
    .stNumberInput input,
    input[type="text"], input[type="password"], input[type="email"], textarea {
        background: white !important;
        border: 1.5px solid #e5e7eb !important;
        border-radius: 10px !important;
        color: #1e1e3a !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus,
    input[type="text"]:focus, input[type="password"]:focus, input[type="email"]:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
        outline: none !important;
    }
    ::placeholder { color: #9ca3af !important; opacity: 1 !important; }

    /* ── Selectbox ── */
    .stSelectbox > div > div, .stSelectbox [data-baseweb="select"] > div {
        background: white !important;
        border: 1.5px solid #e5e7eb !important;
        border-radius: 10px !important; color: #1e1e3a !important;
    }
    [data-baseweb="popover"] > div, [data-baseweb="menu"] {
        background: white !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 10px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important;
    }
    [data-baseweb="option"]:hover { background: #f0f1ff !important; }

    /* ── Cards ── */
    .glass-card {
        background: white;
        border: 1px solid #e8eaf6;
        border-radius: 16px;
        padding: 1.75rem;
        box-shadow: 0 4px 20px rgba(99,102,241,0.07);
        transition: box-shadow 0.2s, border-color 0.2s;
    }
    .glass-card:hover { border-color: #c7d2fe; box-shadow: 0 6px 28px rgba(99,102,241,0.12); }

    /* ── Hero ── */
    .hero { text-align: center; padding: 4rem 1rem 2rem; }
    .hero .badge {
        display: inline-block;
        background: #eef2ff; border: 1px solid #c7d2fe; color: #6366f1;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em;
        text-transform: uppercase; padding: 0.3rem 0.9rem;
        border-radius: 99px; margin-bottom: 1.25rem;
    }
    .hero h1 {
        font-size: clamp(2.5rem, 6vw, 4rem); font-weight: 800;
        letter-spacing: -2px; line-height: 1.1;
        background: linear-gradient(135deg, #4f46e5 20%, #7c3aed);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .hero p { font-size: 1.1rem; color: #6b7280; max-width: 440px; margin: 0 auto; line-height: 1.7; }

    /* ── Divider ── */
    hr { border-color: #e8eaf6 !important; }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        background: white !important;
        border: 1px solid #e8eaf6 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    }

    /* ── Alerts ── */
    [data-testid="stAlert"] { border-radius: 10px !important; }

    /* ── Chat ── */
    [data-testid="stChatMessage"] {
        border-radius: 12px !important;
        border: 1px solid #e8eaf6 !important;
    }

    /* ── Chat Input Bar ── */
    [data-testid="stChatInput"] {
        background: white !important;
        border: 1.5px solid #e5e7eb !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 12px rgba(99,102,241,0.08) !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.15), 0 2px 12px rgba(99,102,241,0.08) !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        border: none !important;
        color: #1e1e3a !important;
        box-shadow: none !important;
    }
    </style>
    """)

def login_register_page():
    inject_css()

    st.html("""
    <div class="hero">
        <div class="badge">✦ AI-Powered Finance</div>
        <h1>MoneyRAG</h1>
        <p>Your personal finance analyst. Upload bank statements, ask questions, get insights — powered by AI.</p>
    </div>
    """)

    col_l, col1, col2, col_r = st.columns([1, 2, 2, 1])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Sign In")
        email = st.text_input("Email", key="login_email", placeholder="you@example.com", label_visibility="collapsed")
        password = st.text_input("Password", type="password", key="login_pass", placeholder="Password", label_visibility="collapsed")
        if st.button("Sign In →", use_container_width=True, type="primary"):
            if email and password:
                with st.spinner(""):
                    try:
                        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                        st.session_state.user = res.user
                        st.session_state.access_token = res.session.access_token
                        st.query_params["t"] = res.session.access_token
                        try:
                            supabase.table("User").upsert({
                                "id": res.user.id,
                                "email": email,
                                "hashed_password": "managed_by_supabase_auth" 
                            }).execute()
                        except Exception as sync_e:
                            print(f"Warning: Could not sync user: {sync_e}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Login failed: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Create Account")
        reg_email = st.text_input("Email", key="reg_email", placeholder="you@example.com", label_visibility="collapsed")
        reg_password = st.text_input("Password", type="password", key="reg_pass", placeholder="Password", label_visibility="collapsed")
        if st.button("Create Account →", use_container_width=True):
            if reg_email and reg_password:
                with st.spinner(""):
                    try:
                        res = supabase.auth.sign_up({"email": reg_email, "password": reg_password})
                        if res.user:
                            try:
                                supabase.table("User").upsert({
                                    "id": res.user.id, "email": reg_email,
                                    "hashed_password": "managed_by_supabase_auth"
                                }).execute()
                            except Exception:
                                pass
                        st.success("Account created! Sign in on the left.")
                    except Exception as e:
                        st.error(f"Signup failed: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    col3, col4, col5 = st.columns(3)
    with col3:
        with st.expander("📚 API Keys"):
            st.markdown("**Google:** [AI Studio](https://aistudio.google.com/app/apikey)")
            st.markdown("**OpenAI:** [Platform](https://platform.openai.com/api-keys)")
    with col4:
        with st.expander("📥 Export Transactions"):
            st.markdown("**Chase:** [Video guide](https://www.youtube.com/watch?v=gtAFaP9Lts8)")
            st.markdown("**Discover:** [Video guide](https://www.youtube.com/watch?v=cry6-H5b0PQ)")
    with col5:
        with st.expander("🏗️ Architecture"):
            st.image("architecture.svg", use_container_width=True)

def load_user_config():
    try:
        # Always get a fresh client with the current auth token
        client = get_supabase()
        res = client.table("AccountConfig").select("*").eq("user_id", st.session_state.user.id).execute()
        if res.data:
            return res.data[0]
    except Exception as e:
        print(f"Failed to load config: {e}")
    return None

def main_app_view():
    inject_css()
    
    # Use session state for active nav tab
    if "nav" not in st.session_state:
        st.session_state.nav = "Chat"

    with st.sidebar:
        st.markdown(f"**MoneyRAG** 💰")
        st.caption(st.session_state.user.email)
        st.divider()
        
        # Modern nav buttons using st.button styled via CSS
        for label, icon in [("Chat", "💬"), ("Ingest Data", "📥"), ("Account Config", "⚙️")]:
            is_active = st.session_state.nav == label
            css_class = "nav-btn-active" if is_active else "nav-btn"
            st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
            if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
                st.session_state.nav = label
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        if st.button("Log Out", use_container_width=True):
            supabase.auth.sign_out()
            if "t" in st.query_params:
                del st.query_params["t"]
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.divider()
        st.caption("[Sajil Awale](https://github.com/AwaleSajil) · [Simran KC](https://github.com/iamsims)")

    nav = st.session_state.nav

    # Always reload config fresh (cached None from unauthenticated loads will persist otherwise)
    config = load_user_config()

    if nav == "Account Config":
        st.header("⚙️ Account Configuration")
        st.write("Configure your AI providers and models here.")
        
        current_provider = config['llm_provider'] if config else "Google"
        current_key = config['api_key'] if config else ""
        current_decode = config.get('decode_model', "gemini-3-flash-preview") if config else "gemini-3-flash-preview"
        current_embed = config.get('embedding_model', "gemini-embedding-001") if config else "gemini-embedding-001"
        # Provider Selection - Default to Google
        provider = st.selectbox("LLM Provider", ["Google", "OpenAI"], index=0 if (not config or config['llm_provider'] == "Google") else 1)
        
        if provider == "Google":
            models = ["gemini-3-flash-preview", "gemini-3-pro-image-preview", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
            embeddings = ["gemini-embedding-001"]
        else:
            models = ["gpt-5-mini", "gpt-5-nano", "gpt-4o-mini", "gpt-4o"]
            embeddings = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]

        with st.form("config_form"):
            api_key = st.text_input("API Key", type="password", value=current_key)
            
            col1, col2 = st.columns(2)
            with col1:
                # Default to gemini-3 if no config exists
                m_default_val = current_decode if config else "gemini-3-flash-preview"
                m_idx = models.index(m_default_val) if m_default_val in models else 0
                final_decode = st.selectbox("Select Model", models, index=m_idx)

            with col2:
                e_idx = embeddings.index(current_embed) if (config and current_embed in embeddings) else 0
                final_embed = st.selectbox("Select Embedding Model", embeddings, index=e_idx)
            
            submitted = st.form_submit_button("Save Configuration", type="primary", use_container_width=True)
            if submitted:
                if not api_key:
                    st.error("API Key is required.")
                else:
                    try:
                        record = {
                            "user_id": st.session_state.user.id,
                            "llm_provider": provider,
                            "api_key": api_key,
                            "decode_model": final_decode,
                            "embedding_model": final_embed
                        }
                        if config:
                            supabase.table("AccountConfig").update(record).eq("id", config['id']).execute()
                        else:
                            supabase.table("AccountConfig").insert(record).execute()
                        
                        st.session_state.user_config = load_user_config()
                        # Reinitialize RAG with new config
                        if "rag" in st.session_state:
                            del st.session_state.rag
                            
                        st.success("Configuration saved successfully!")
                    except Exception as e:
                        st.error(f"Failed to save configuration: {e}")

    elif nav == "Ingest Data":
        st.header("📥 Ingest Data")
        
        uploaded_files = st.file_uploader("Upload CSV transactions or Receipt Images", accept_multiple_files=True, type=['csv', 'png', 'jpg', 'jpeg'])
        if uploaded_files:
            if st.button("Ingest Selected Files", type="primary"):
                if not config:
                    st.error("Please set up your Account Config first!")
                    return
                
                # Initialize RAG if needed
                if "rag" not in st.session_state:
                    st.session_state.rag = MoneyRAG(
                        llm_provider=config["llm_provider"], 
                        model_name=config.get("decode_model", "gemini-2.5-pro"), 
                        embedding_model_name=config.get("embedding_model", "gemini-embedding-001"), 
                        api_key=config["api_key"],
                        user_id=st.session_state.user.id,
                        access_token=st.session_state.access_token
                    )

                uploaded_files_info = []
                user_id = st.session_state.user.id
                
                with st.spinner("Uploading to Supabase Storage & Processing..."):
                    for uploaded_file in uploaded_files:
                        # 1. Save temp locally for parsing
                        local_path = os.path.join(st.session_state.rag.temp_dir, uploaded_file.name)
                        with open(local_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # 2. Upload raw file to Supabase Object Storage
                        is_image = uploaded_file.name.lower().endswith(('.png', '.jpg', '.jpeg'))
                        folder = "bills" if is_image else "csvs"
                        s3_key = f"{user_id}/{folder}/{uploaded_file.name}"
                        content_type = "image/jpeg" if is_image else "text/csv"
                        if uploaded_file.name.lower().endswith('.png'):
                            content_type = "image/png"

                        try:
                            supabase.storage.from_("money-rag-files").upload(
                                file=local_path,
                                path=s3_key,
                                file_options={"content-type": content_type, "upsert": "true"}
                            )
                            
                            # 3. Log the upload in the correct table depending on type
                            if is_image:
                                file_record = supabase.table("BillFile").insert({
                                    "user_id": user_id,
                                    "filename": uploaded_file.name,
                                    "s3_key": s3_key
                                }).execute()
                            else:
                                file_record = supabase.table("CSVFile").insert({
                                    "user_id": user_id,
                                    "filename": uploaded_file.name,
                                    "s3_key": s3_key
                                }).execute()
                                
                            file_id = file_record.data[0]['id']
                            uploaded_files_info.append({"path": local_path, "file_id": file_id})
                            
                        except Exception as e:
                            st.error(f"Error uploading {uploaded_file.name}: {e}")
                            continue

                    # 4. Trigger the parsing, routing data to Supabase Postgres
                    if uploaded_files_info:
                        asyncio.run(st.session_state.rag.setup_session(uploaded_files_info))
                        st.success("Data uploaded, parsed, and vectorized securely!")
                        st.rerun()

        st.divider()
        st.subheader("Your Uploaded Files")
        try:
            res_csv = supabase.table("CSVFile").select("*").eq("user_id", st.session_state.user.id).execute()
            res_bill = supabase.table("BillFile").select("*").eq("user_id", st.session_state.user.id).execute()
            
            files = []
            if res_csv.data:
                for d in res_csv.data: d['type'] = 'csv'
                files.extend(res_csv.data)
            if res_bill.data:
                for d in res_bill.data: d['type'] = 'bill'
                files.extend(res_bill.data)
                
            if not files:
                st.info("No files uploaded yet.")
            else:
                for f in files:
                    col_file, col_del = st.columns([4, 1])
                    with col_file:
                        st.write(f"📄 **{f['filename']}** (Uploaded: {f['upload_date'][:10]})")
                    with col_del:
                        if st.button("Delete", key=f"del_{f['id']}"):
                            st.session_state[f"confirm_del_{f['id']}"] = True

                        if st.session_state.get(f"confirm_del_{f['id']}", False):
                            st.warning("Are you sure? This permanently deletes the file from Cloud Storage, the SQL Database, and the Vector Index.")
                            col_y, col_n = st.columns(2)
                            with col_y:
                                if st.button("Yes, Delete", key=f"yes_{f['id']}", type="primary"):
                                    with st.spinner("Purging file data..."):
                                        try:
                                            # Delete from storage
                                            supabase.storage.from_("money-rag-files").remove([f['s3_key']])
                                        except Exception as e:
                                            print(f"Warning storage delete failed: {e}")
                                            
                                        # Use initialized RAG to delete from Vectors and Postgres
                                        if "rag" not in st.session_state and config:
                                            st.session_state.rag = MoneyRAG(
                                                llm_provider=config["llm_provider"], 
                                                model_name=config.get("decode_model", "gemini-2.5-pro"), 
                                                embedding_model_name=config.get("embedding_model", "gemini-embedding-001"), 
                                                api_key=config["api_key"],
                                                user_id=st.session_state.user.id,
                                                access_token=st.session_state.access_token
                                            )
                                        if "rag" in st.session_state:
                                            asyncio.run(st.session_state.rag.delete_file(f['id'], f['type']))
                                        else:
                                            table = "CSVFile" if f['type'] == 'csv' else "BillFile"
                                            # Fallback if no RAG config
                                            if f['type'] == 'csv':
                                                supabase.table("Transaction").delete().eq("source_csv_id", f['id']).execute()
                                            supabase.table(table).delete().eq("id", f['id']).execute()
                                            
                                    del st.session_state[f"confirm_del_{f['id']}"]
                                    st.success(f"Deleted {f['filename']}!")
                                    st.rerun()
                                    
                            with col_n:
                                if st.button("Cancel", key=f"cancel_{f['id']}"):
                                    del st.session_state[f"confirm_del_{f['id']}"]
                                    st.rerun()
                            
        except Exception as e:
            st.error(f"Failed to load files: {e}")

    elif nav == "Chat":
        st.header("💬 Financial Assistant")
        if not config:
            st.warning("Please configure your Account Config (API Key) first!")
            return
            
        if "rag" not in st.session_state:
            st.session_state.rag = MoneyRAG(
                llm_provider=config["llm_provider"], 
                model_name=config.get("decode_model", "gemini-2.5-pro"), 
                embedding_model_name=config.get("embedding_model", "gemini-embedding-001"), 
                api_key=config["api_key"],
                user_id=st.session_state.user.id,
                access_token=st.session_state.access_token
            )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Show file ingestion status
        try:
            client = get_supabase()
            files_csv = client.table("CSVFile").select("id, filename").eq("user_id", st.session_state.user.id).execute()
            files_bill = client.table("BillFile").select("id, filename").eq("user_id", st.session_state.user.id).execute()
            
            all_files = (files_csv.data or []) + (files_bill.data or [])
            file_count = len(all_files)
            
            if file_count == 0:
                st.warning("⚠️ No data loaded yet. Go to **Ingest Data** to upload a CSV or Bill file before chatting.")
            else:
                names = ", ".join(f['filename'] for f in all_files[:3])
                suffix = f" + {file_count - 3} more" if file_count > 3 else ""
                st.info(f"📊 **{file_count} file{'s' if file_count > 1 else ''} loaded:** {names}{suffix}")
        except Exception:
            pass  # Don't break chat if the status check fails


        # Helper function to cleverly render either text or a Plotly chart
        def render_content(content):
            if isinstance(content, str) and "===CHART===" in content:
                parts = content.split("===CHART===")
                st.markdown(parts[0].strip())
                
                for part in parts[1:]:
                    if "===ENDCHART===" in part:
                        chart_json, remaining_text = part.split("===ENDCHART===")
                        try:
                            fig = pio.from_json(chart_json.strip())
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error("Failed to render chart.")
                        
                        if remaining_text.strip():
                            st.markdown(remaining_text.strip())
            else:
                st.markdown(content)

        # Render previous messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                render_content(message["content"])

        # Handle new user input
        if prompt := st.chat_input("Ask about your spending..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    # Drain the async generator, collecting all events first
                    async def run_chat_stream():
                        events = []
                        async for event in st.session_state.rag.chat(prompt):
                            events.append(event)
                        return events

                    with st.spinner("Thinking..."):
                        events = asyncio.run(run_chat_stream())

                    # Show tool-call trace in a collapsible expander
                    tool_events = [e for e in events if e["type"] in ("tool_start", "tool_end")]
                    final_event = next((e for e in events if e["type"] == "final"), None)

                    if tool_events:
                        tool_names = list(dict.fromkeys(
                            e["name"].replace("money_rag_", "").replace("_", " ").title()
                            for e in tool_events if e["type"] == "tool_start"
                        ))
                        with st.expander(f"🔍 Used: {', '.join(tool_names)}", expanded=False):
                            for e in tool_events:
                                if e["type"] == "tool_start":
                                    st.markdown(f"**▶ `{e['name']}`**")
                                    if e.get("input"):
                                        st.caption(f"Input: {e['input']}")
                                else:
                                    st.caption(f"↳ {e.get('snippet', '')}")

                    response = final_event["content"] if final_event else "No response."
                    render_content(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error during chat: {e}")

if __name__ == "__main__":
    # Attempt to restore session from query params if page was refreshed
    if "user" not in st.session_state:
        token_from_url = st.query_params.get("t")
        if token_from_url:
            try:
                res = supabase.auth.get_user(token_from_url)
                if res and res.user:
                    st.session_state.user = res.user
                    st.session_state.access_token = token_from_url
            except Exception:
                # Token is invalid/expired - clear it from the URL too
                if "t" in st.query_params:
                    del st.query_params["t"]

    if "user" not in st.session_state:
        login_register_page()
    else:
        main_app_view()