# --- Standard library imports ---
import json
import logging
import os
import sys
import time
import uuid

# --- Third-party imports ---
import requests
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# --- Local application imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from chatbot.chatbot import app
from chatbot.tools import initialize_vector_store
from utils.api_keys import validate_api_keys

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://0.0.0.0:8181"

def is_api_server_running():
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False   


initialize_vector_store()

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="RAG - LLM app", 
    page_icon="📚", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# --- API Key Validation ---
keys = validate_api_keys()

# --- Header ---
st.html("""<h2 style="text-align: center;">📚🔍 <i> Do your LLM even RAG bro? </i> 🤖💬</h2>""")


# --- Initial Setup ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="👋🤖 RAG-bot na área! No que posso te ajudar hoje?")  # Default welcome message
    ]
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'language_model' not in st.session_state:
    st.session_state.language_model = "openai/gpt-4o-mini"
if "response_time_overall" not in st.session_state:
    st.session_state.response_time_overall = 0.0
if "response_time_process" not in st.session_state:
    st.session_state.response_time_process = 0.0
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# --- Sidebar ---
MODELS = [
    "openai/gpt-4o-mini",
    "groq/llama-3.3-70b-versatile"
]

with st.sidebar:
    st.divider()

    if 'previous_model' not in st.session_state:
        st.session_state.previous_model = MODELS[0] if MODELS else None
    
    def on_model_change():
        if st.session_state.language_model != st.session_state.previous_model:
            st.session_state.messages = [AIMessage(content="👋🤖 RAG-bot na área! No que posso te ajudar hoje?")] # clear the chat history
            # Show toast notification
            st.toast(f"✅ Model changed to: {st.session_state.language_model}", icon="🤖")
            st.session_state.previous_model = st.session_state.language_model

    st.selectbox(
        "🤖 Selecione um modelo:", 
        options=MODELS,
        key="language_model",
        on_change=on_model_change
    )

    st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")
    st.header("RAG Recursos:")
        
    # File upload input for RAG with documents
    uploaded_files = st.file_uploader(
        "📄 Carregue seus arquivos (PDF, imagens para OCR)", 
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="rag_docs",
    )

    st.divider()
    st.write("📋[README](add here)")

# --- File Upload and Processing ---
if uploaded_files:
    if not is_api_server_running():
        st.error("Document processing server is not running. Please start the API server on port 8181.")
    else:
        files = []
        for uploaded_file in uploaded_files:
            files.append((
                'files', 
                (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            ))

        response = requests.post(
            f"{BASE_URL}/documents",
            files=files
        )

        if response.status_code == 200:
            st.success("Files processed successfully!")
            st.json(response.json())
    
# Display existing chat history first
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)
    elif isinstance(msg, ToolMessage):
        with st.chat_message("tool", avatar="📚"):
            pass

# --- User input ---
user_input = st.chat_input("Digite aqui...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Generate response
    with st.spinner("Generating response..."):
        start_time = time.time()

        response = app.invoke({'messages': st.session_state.messages})
        response_time = time.time() - start_time

    # Check the response structure explicitly
    for msg in response["messages"][len(st.session_state.messages):]:
        if isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)
                st.caption(f"⏱️ {response_time:.2f}s")   
            st.session_state.messages.append(msg)

        elif isinstance(msg, ToolMessage):
            with st.chat_message("tool", avatar="📚"):
                st.markdown(msg.content)
                try:
                    tool_data = json.loads(msg.content)
                    sources = []
                    if "metadatas" in tool_data and "documents" in tool_data:
                        for meta_group, doc_group in zip(tool_data["metadatas"], tool_data["documents"]):
                            for metadata, document in zip(meta_group, doc_group):
                                sources.append({
                                    "source": metadata.get("source", "Sem fonte."),
                                    "page": metadata.get("page", "Sem página."),
                                    "type": metadata.get("type", "Sem tipo de documento."),
                                    "tamanho_trecho": len(document),
                                    "content_sample": document[:200] + "..." 
                                })
                    if sources:
                        st.markdown("Documentos encontrados:")
                        for i, source in enumerate(sources, 1):
                            with st.expander(f"Documento {i}"):
                                st.markdown(f"""
                                - *Arquivo*: **"{source['source']}"**
                                - *Página*: {source['page']}
                                - *Tipo*: ".{source['type']}"
                                - *Tamanho do Trecho*: {source['tamanho_trecho']}
                                - *Trecho*: `{source['content_sample']}`
                                """)
                    else:
                        st.warning("Nenhum documento relevante encontrado.")
                except json.JSONDecodeError:
                    st.error("Erro ao processar resposta da ferramenta")
                    st.markdown(msg.content)
            
            st.session_state.messages.append(msg)