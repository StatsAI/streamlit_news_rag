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
from langchain_ollama import ChatOllama
from langchain.chains.summarize import load_summarize_chain
import chromadb
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

# --- Page Config & Styling ---
st.set_page_config(page_title="CNN News Intelligence", layout="wide")

try:
    logo = Image.open('images/picture.png')
    st.markdown(
        """
        <style>
            [data-testid=stSidebar] [data-testid=stImage]{
                text-align: center;
                display: block;
                margin-left: auto;
                margin-right: auto;
                margin-top: -25px;
                width: 100%;
            }
            .block-container { padding-top: 1rem; }
        </style>
        """, unsafe_allow_html=True
    )
    with st.sidebar:
        st.image(logo)
except FileNotFoundError:
    pass

# --- Backend Functions ---

@st.cache_resource()
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_data(ttl="1d")
def pull_latest_links():
    cnn_lite_url = "https://lite.cnn.com/"
    try:
        elements = partition_html(url=cnn_lite_url)
        links = []
        for element in elements:
            if element.metadata.link_urls:
                rel = element.metadata.link_urls[0]
                full_url = rel if rel.startswith('http') else f"{cnn_lite_url}{rel}"
                links.append(full_url)
        return links[1:-2] if len(links) > 2 else links
    except Exception as e:
        st.error(f"Scraping Error: {e}")
        return []

@st.cache_resource(ttl="1d")
def load_vector_database(_embedding_func, _docs):
    if not _docs: return None
    client = chromadb.PersistentClient(path=".chromadb")
    try:
        client.delete_collection("cnn_docs")
    except:
        pass 
    return Chroma.from_documents(documents=_docs, embedding=_embedding_func, 
                                 collection_name="cnn_docs", client=client)

@st.cache_resource(ttl="1d")
def load_docs_parallel(urls):
    with ThreadPoolExecutor() as executor:
        loaders = [UnstructuredURLLoader(urls=[u]) for u in urls]
        docs_list = list(executor.map(lambda l: l.load(), loaders))
    return [doc for sublist in docs_list for doc in sublist]

# --- Hybrid LLM Logic ---

@st.cache_resource
def get_gemini():
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", # Latest 2026 Free Tier Model
            google_api_key=st.secrets["gemini_api_secret_name"],
            temperature=0.7
        )
    except: return None

@st.cache_resource
def get_local_phi():
    # Requires 'ollama run phi4-mini' to be active locally
    return ChatOllama(model="phi4-mini", temperature=0.7)

def run_hybrid_summarization(relevant_docs):
    """Try Gemini first (1 request for all docs), fallback to Phi-4 if 429 occurs."""
    gemini = get_gemini()
    if gemini:
        try:
            chain = load_summarize_chain(gemini, chain_type="stuff")
            res = chain.invoke({"input_documents": relevant_docs})
            return res['output_text'], "Gemini 2.5 Cloud"
        except Exception as e:
            if "429" in str(e):
                st.warning("Cloud Quota Exceeded. Falling back to Local Phi-4 Mini...")
            else:
                st.error(f"Cloud Error: {e}")
    
    # Fallback to local
    try:
        local_llm = get_local_phi()
        chain = load_summarize_chain(local_llm, chain_type="stuff")
        res = chain.invoke({"input_documents": relevant_docs})
        return res['output_text'], "Local Phi-4 Mini-Flash"
    except Exception as e:
        return f"Error: Both models failed. Is Ollama running? ({e})", "None"

# --- Main App Execution ---

st.title("CNN RAG Intelligence")
st.info("Uses Gemini 2.5 (Cloud) with local Phi-4 (Ollama) as a fallback.")

with st.status("Fetching latest news...", expanded=False) as status:
    links = pull_latest_links()
    docs = load_docs_parallel(links)
    vectorstore = load_vector_database(load_embedding_model(), docs)
    status.update(label="System Ready!", state="complete")

with st.sidebar:
    query = st.text_input("Search Topic:", value="Global Economy")
    run_button = st.button('Generate Summaries')

if (run_button or (query and query != st.session_state.get('last_query', ""))) and vectorstore:
    st.session_state['last_query'] = query
    
    with st.spinner(f"Analyzing articles for '{query}'..."):
        relevant_docs = vectorstore.similarity_search(query, k=5)
        
        if not relevant_docs:
            st.warning("No relevant articles found.")
        else:
            summary, model_name = run_hybrid_summarization(relevant_docs)
            
            st.subheader(f"Analysis via {model_name}")
            st.markdown(summary)
            
            with st.expander("Sources Cited"):
                for d in relevant_docs:
                    st.caption(d.metadata.get('source', 'CNN'))
