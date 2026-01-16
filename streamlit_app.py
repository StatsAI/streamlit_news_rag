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
from langchain.chains.summarize import load_summarize_chain
import chromadb
from chromadb.config import Settings
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

# --- Page Config & Styling ---
st.set_page_config(page_title="CNN Summarizer", layout="wide")

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
    st.sidebar.warning("Logo not found. Please check 'images/picture.png' path.")

# --- Functions ---

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
                relative_link = element.metadata.link_urls[0]
                # Ensure we don't double up the URL if it's already absolute
                full_url = relative_link if relative_link.startswith('http') else f"{cnn_lite_url}{relative_link}"
                links.append(full_url)
        return links[1:-2] if len(links) > 2 else links
    except Exception as e:
        st.error(f"Failed to pull links: {e}")
        return []

@st.cache_resource(ttl="1d")
def get_chroma_client():
    return chromadb.PersistentClient(path=".chromadb")

@st.cache_resource(ttl="1d")
def load_vector_database(_embedding_function, _docs):
    if not _docs:
        return None
        
    chroma_client = get_chroma_client()
    collection_name = "cnn_doc_embeddings"
    
    # Safely clear old collection to avoid schema mismatch or stale data
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass 
    
    return Chroma.from_documents(
        documents=_docs, 
        embedding=_embedding_function, 
        collection_name=collection_name, 
        client=chroma_client
    )

@st.cache_resource(ttl="1d")
def load_documents_parallel(urls):
    if not urls:
        return []
    with ThreadPoolExecutor() as executor:
        loaders = [UnstructuredURLLoader(urls=[url]) for url in urls]
        docs_list = list(executor.map(lambda loader: loader.load(), loaders))
    return [doc for sublist in docs_list for doc in sublist]

@st.cache_resource
def load_gemini_model():    
    gem_api_key = st.secrets["gemini_api_secret_name"]
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        google_api_key=gem_api_key
    )

# --- Main Logic ---

st.title("CNN Article Summarization")
st.write("Ingesting latest CNN articles into a RAG pipeline for on-demand summarization.")

# 1. Scrape & Process
with st.status("Initializing Data...", expanded=False) as status:
    links = pull_latest_links()
    st.write(f"Found {len(links)} articles.")
    
    docs = load_documents_parallel(links)
    st.write(f"Extracted content from {len(docs)} pages.")
    
    embedding_function = load_embedding_model()
    
    vectorstore = load_vector_database(embedding_function, docs)
    
    llm = load_gemini_model()
    status.update(label="System Ready!", state="complete")

# 2. Sidebar Input
with st.sidebar:
    query = st.text_input("Topic Selection", value="Trump", key="query_text")
    run_button = st.button('Summarize Articles')

# 3. Execution
if 'last_query' not in st.session_state:
    st.session_state['last_query'] = ""

# Trigger if button pressed OR if a new query is entered
if (run_button or (query and query != st.session_state['last_query'])) and vectorstore:
    st.session_state['last_query'] = query
    
    with st.spinner(f"Searching for '{query}'..."):
        query_docs = vectorstore.similarity_search(query, k=5)
        chain = load_summarize_chain(llm, chain_type="stuff")

        if not query_docs:
            st.warning("No relevant articles found for that topic.")
        else:
            for doc in query_docs:
                with st.container():
                    res = chain.invoke({"input_documents": [doc]})
                    st.markdown(f"### Summary")
                    st.write(res['output_text'])
                    st.caption(f"**Source:** {doc.metadata.get('source', 'CNN Lite')}")
                    st.divider()

# 4. Cache Management
st.sidebar.markdown("---")
if st.sidebar.button('Clear Cache & Refresh News'):
    pull_latest_links.clear()
    load_documents_parallel.clear()
    load_vector_database.clear()
    st.cache_resource.clear()
    st.rerun()
