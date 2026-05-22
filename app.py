import os
import streamlit as st
from dotenv import load_dotenv
import rag_pipeline

# Page Configuration
st.set_page_config(
    page_title="Ultra-RAG PDF Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Injection of custom premium styles and fonts
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Apply globally */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Modern neon title styling */
    .title-container {
        padding: 1.5rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(167, 139, 250, 0.15) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        backdrop-filter: blur(12px);
    }
    
    .main-title {
        background: linear-gradient(135deg, #a78bfa 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    /* Document Item Cards */
    .doc-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .doc-name {
        color: #e2e8f0;
        font-size: 0.95rem;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 80%;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(167, 139, 250, 0.2);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(167, 139, 250, 0.4);
    }
    
    /* Empty State Splash Container */
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        background: rgba(30, 41, 59, 0.3);
        border: 2px dashed rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        margin-top: 2rem;
    }
    
    .empty-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #a78bfa 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE SETUP -----------------

# Auto-sync indexed files from persistent ChromaDB on startup
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = rag_pipeline.get_indexed_pdfs()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------- SIDEBAR WORKSPACE -----------------

st.sidebar.markdown("## ⚙️ Configuration")

# Handle OpenRouter API Key securely
api_key = os.getenv("OPENROUTER_API_KEY", "")
if not api_key:
    # If not in environment, allow entering it in the sidebar
    api_key = st.sidebar.text_input(
        "OpenRouter API Key",
        type="password",
        placeholder="sk-or-v1-...",
        help="Get your API key at openrouter.ai. It will not be stored permanently."
    )
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
else:
    st.sidebar.success("🔑 OpenRouter API Key detected from `.env`")

st.sidebar.markdown("---")
st.sidebar.markdown("## 📂 Document Hub")

# Multi-PDF File Uploader
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True,
    help="Support scanning, chunking and local vector embedding indexing."
)

# Index new uploads
if uploaded_files:
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        if filename not in st.session_state.indexed_files:
            with st.sidebar.status(f"Parsing & indexing `{filename}`...", expanded=True) as status:
                try:
                    success = rag_pipeline.ingest_pdf(uploaded_file, filename)
                    if success:
                        st.session_state.indexed_files.append(filename)
                        status.update(label=f"✓ Indexed: {filename}", state="complete")
                except Exception as e:
                    status.update(label=f"✗ Failed: {filename}", state="error")
                    st.sidebar.error(f"Error parsing `{filename}`: {e}")

st.sidebar.markdown("### 📚 Indexed Documents")

# Display indexed files with individual remove buttons
if st.session_state.indexed_files:
    for filename in list(st.session_state.indexed_files):
        col_name, col_btn = st.sidebar.columns([0.85, 0.15])
        with col_name:
            st.markdown(f"""
            <div class="doc-card">
                <span class="doc-name" title="{filename}">📄 {filename}</span>
            </div>
            """, unsafe_allow_html=True)
        with col_btn:
            # Subtle delete button aligning with the card
            if st.button("🗑️", key=f"del_{filename}", help=f"Remove '{filename}' from index"):
                with st.spinner(f"Removing {filename}..."):
                    rag_pipeline.delete_pdf(filename)
                    st.session_state.indexed_files.remove(filename)
                    st.toast(f"Removed '{filename}' from database.")
                    st.rerun()
else:
    st.sidebar.info("No documents indexed yet.")

# Clear All Documents Button
st.sidebar.markdown("---")
if st.session_state.indexed_files:
    if st.sidebar.button("🗑️ Clear All Documents", type="primary", use_container_width=True):
        with st.spinner("Wiping local ChromaDB collection..."):
            rag_pipeline.delete_all()
            st.session_state.indexed_files = []
            st.session_state.messages = []
            st.toast("Successfully deleted all documents & cleared history.")
            st.rerun()

# ----------------- MAIN APP WORKSPACE -----------------

# Header Section
st.markdown("""
<div class="title-container">
    <h1 class="main-title">AI RAG PDF Chatbot</h1>
    <p class="subtitle">Chat with multiple documents simultaneously using local HuggingFace embeddings and Gemini 2.0 Flash</p>
</div>
""", unsafe_allow_html=True)

# Handle Empty State (No PDFs uploaded)
if not st.session_state.indexed_files:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">📚</div>
        <h2>Welcome to RAG Chatbot Workspace</h2>
        <p style="color: #94a3b8; max-width: 600px; margin: 0 auto 1.5rem auto;">
            To begin, upload your PDF files in the sidebar. We will chunk their contents, 
            generate semantic vectors using <b>sentence-transformers</b> locally, and store them in 
            your local <b>ChromaDB</b> vector database.
        </p>
        <p style="font-size: 0.9rem; color: #a78bfa;">⚡ Embeddings run 100% locally on your machine!</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Display source citations if they exist
            if msg.get("sources"):
                with st.expander("🔍 View Reference Passages", expanded=False):
                    for idx, doc in enumerate(msg["sources"]):
                        source = doc.get("source", "Unknown PDF")
                        page = doc.get("page", "N/A")
                        content = doc.get("content", "")
                        st.markdown(f"**Reference {idx+1}:** `{source}` — Page {page}")
                        st.info(content)

    # Chat Input Field
    if prompt := st.chat_input("Ask a question about your indexed documents..."):
        
        # Verify OpenRouter key before query
        if not os.getenv("OPENROUTER_API_KEY"):
            st.error("Missing OpenRouter API Key. Please add it to your `.env` or input it in the sidebar.")
        else:
            # 1. Show and append User Query
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # 2. Get answer from LangChain Pipeline
            with st.chat_message("assistant"):
                with st.spinner("Analyzing context & generating response..."):
                    try:
                        # Construct chat history in (User, Assistant) tuples format for LangChain
                        chat_history = []
                        temp_user = None
                        for msg in st.session_state.messages[:-1]: # Exclude the query we just added
                            if msg["role"] == "user":
                                temp_user = msg["content"]
                            elif msg["role"] == "assistant" and temp_user is not None:
                                chat_history.append((temp_user, msg["content"]))
                                temp_user = None

                        # Call the pipeline
                        result = rag_pipeline.get_answer(prompt, chat_history)
                        answer = result["answer"]
                        source_docs = result["source_documents"]
                        
                        # Render answer
                        st.markdown(answer)
                        
                        # Parse source documents for persistence and rendering
                        parsed_sources = []
                        for doc in source_docs:
                            parsed_sources.append({
                                "source": doc.metadata.get("source", "Unknown"),
                                "page": doc.metadata.get("page", "N/A"),
                                "content": doc.page_content
                            })
                            
                        # Render reference expander
                        if parsed_sources:
                            with st.expander("🔍 View Reference Passages", expanded=False):
                                for idx, src in enumerate(parsed_sources):
                                    st.markdown(f"**Reference {idx+1}:** `{src['source']}` — Page {src['page']}")
                                    st.info(src["content"])
                                    
                        # Save answer and source structures to session state for persistence
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": parsed_sources
                        })
                        
                    except Exception as e:
                        err_msg = f"An error occurred during retrieval/chat generation: {e}"
                        st.error(err_msg)
                        # Remove last user message on failure to keep history clean/synchronized
                        st.session_state.messages.pop()
