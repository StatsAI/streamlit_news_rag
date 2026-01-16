__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3
import time

import streamlit as st
from unstructured.partition.html import partition_html
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama  # Local model integration
from langchain.chains.summarize import load_summarize_chain
import chromadb
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

# --- Page Config & Styling ---
st.set_page_config(page_title="CNN Summarizer (Hybrid RAG)", layout="wide")

# ... [Logo styling remains the same as your original code] ...

# --- Robust Model Loading ---

@st.cache_resource
def load_gemini_model():    
    try:
        gem_api_key = st.secrets["gemini_api_secret_name"]
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.7,
            google_api_key=gem_api_key
        )
    except:
        return None

@st.cache_resource
def load_local_model():
    """
    Attempts to connect to Phi-4 Mini-Flash via Ollama.
    Ensure you have run 'ollama run phi4-mini' in your terminal first.
    """
    return ChatOllama(
        model="phi4-mini", 
        temperature=0.7
    )

def run_summarization(docs, query):
    """
    Tries Gemini first, falls back to Phi-4 if quota is hit.
    """
    gemini_llm = load_gemini_model()
    
    # Strategy 1: Attempt Gemini
    if gemini_llm:
        try:
            chain = load_summarize_chain(gemini_llm, chain_type="stuff")
            return chain.invoke({"input_documents": docs})['output_text'], "Gemini 2.5 Flash-Lite"
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                st.warning("Gemini Quota Exceeded. Switching to local Phi-4 Mini...")
            else:
                st.error(f"Gemini error: {e}")
    
    # Strategy 2: Fallback to Local Phi-4
    try:
        local_llm = load_local_model()
        chain = load_summarize_chain(local_llm, chain_type="stuff")
        return chain.invoke({"input_documents": docs})['output_text'], "Local Phi-4 Mini-Flash"
    except Exception as e:
        return f"Error: Both models failed. (Local Error: {e})", "None"

# ... [Data processing functions: pull_latest_links, load_vector_database, etc. remain the same] ...

# --- Main Logic ---

st.title("CNN Article Summarization")
st.caption("Hybrid Mode: Uses Cloud Gemini primarily, falls back to Local Phi-4 if offline or over quota.")

# ... [Initialization status bar remains the same] ...

# 3. Execution Logic
if (run_button or (query and query != st.session_state.get('last_query', ""))) and vectorstore:
    st.session_state['last_query'] = query
    
    with st.spinner(f"Analyzing articles for '{query}'..."):
        query_docs = vectorstore.similarity_search(query, k=5)

        if not query_docs:
            st.warning("No relevant articles found.")
        else:
            # We process all docs through the fallback logic
            summary, model_used = run_summarization(query_docs, query)
            
            st.markdown(f"### Summary ({model_used})")
            st.write(summary)
            
            with st.expander("View Sources"):
                for doc in query_docs:
                    st.caption(f"Source: {doc.metadata.get('source')}")
                    st.divider()
