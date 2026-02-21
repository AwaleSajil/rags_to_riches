import streamlit as st
import asyncio
import os
from money_rag import MoneyRAG

st.set_page_config(page_title="MoneyRAG", layout="wide")

# Sidebar for Authentication
with st.sidebar:
    st.header("Authentication")
    provider = st.selectbox("LLM Provider", ["Google", "OpenAI"])
    
    if provider == "Google":
        models = ["gemini-3-flash-preview", "gemini-3-pro-image-preview", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
        embeddings = ["text-embedding-004"]
    else:
        models = ["gpt-5-mini", "gpt-5-nano", "gpt-4o-mini", "gpt-4o"]
        embeddings = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
        
    model_name = st.selectbox("Choose Decoder Model", models)
    embed_name = st.selectbox("Choose Embedding Model", embeddings)
    api_key = st.text_input("API Key", type="password")
    
    auth_button = st.button("Authenticate")
    if auth_button and api_key:
        st.session_state.rag = MoneyRAG(provider, model_name, embed_name, api_key)
        st.success("Authenticated!")

# Main Window
st.title("MoneyRAG 💰")
st.subheader("Where is my money?")
st.markdown("""
This app helps you analyze your personal finances using AI. 
Upload your bank/credit card CSV statements to chat with your data semantically.
""")

# Guides Section
col1, col2 = st.columns(2)

with col1:
    with st.expander("📚 How to get API keys"):
        st.markdown("**Google Gemini API:**")
        st.markdown("🔗 [Get API key from Google AI Studio](https://aistudio.google.com/app/apikey)")
        st.markdown("")
        st.markdown("**OpenAI API:**")
        st.markdown("🔗 [Get API key from OpenAI Platform](https://platform.openai.com/api-keys)")

with col2:
    with st.expander("📥 How to download transaction history"):
        st.markdown("**Chase Credit Card:**")
        st.video("https://www.youtube.com/watch?v=gtAFaP9Lts8")
        st.markdown("")
        st.markdown("**Discover Credit Card:**")
        st.video("https://www.youtube.com/watch?v=cry6-H5b0PQ")

# Architecture Diagram
with st.expander("🏗️ How MoneyRAG Works"):
    st.image("architecture.svg", use_container_width=True)

st.divider()

if "rag" in st.session_state:
    uploaded_files = st.file_uploader("Upload CSV transactions", accept_multiple_files=True, type=['csv'])
    
    if uploaded_files:
        if st.button("Ingest Data"):
            temp_paths = []
            for uploaded_file in uploaded_files:
                path = os.path.join(st.session_state.rag.temp_dir, uploaded_file.name)
                with open(path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                temp_paths.append(path)
            
            with st.spinner("Ingesting and vectorizing..."):
                asyncio.run(st.session_state.rag.setup_session(temp_paths))
            st.success("Data ready for chat!")

    # Chat Interface
    st.divider()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your spending..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = asyncio.run(st.session_state.rag.chat(prompt))
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Please authenticate in the sidebar to start.")