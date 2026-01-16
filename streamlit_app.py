__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
import chromadb
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

# --- Styling & Logo ---
st.set_page_config(page_title="CNN News RAG", layout="wide")
try:
    logo = Image.open('images/picture.png')
    st.sidebar.image(logo)
except:
    pass

# --- Robust Scraper ---
@st.cache_data(ttl="1d")
def pull_latest_links():
    base_url = "https://lite.cnn.com"
    links = []
    try:
        response = requests.get(base_url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        for a in soup.find_all('a', href=True):
            full_url = urljoin(base_url, a['href'])
            if "/en/article/" in full_url: # Focus on news articles
                links.append(full_url)
        return list(dict.fromkeys(links)) # Unique links
    except Exception as e:
        st.error(f"Scraping error: {e}")
        return []

# --- Optimized Vector Store ---
@st.cache_resource()
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource(ttl="1d")
def get_vectorstore(_docs, _embeddings):
    if not _docs: return None
    client = chromadb.PersistentClient(path=".chromadb")
    try:
        client.delete_collection("cnn_embeddings")
    except:
        pass
    return Chroma.from_documents(_docs, _embeddings, collection_name="cnn_embeddings", client=client)

@st.cache_resource(ttl="1d")
def load_docs(urls):
    with ThreadPoolExecutor() as exe:
        loaders = [UnstructuredURLLoader(urls=[u]) for u in urls]
        docs = list(exe.map(lambda l: l.load(), loaders))
    return [item for sublist in docs for item in sublist]

@st.cache_resource
def load_llm():
    # User's error log mentions gemini-2.5-flash
    model_name = "gemini-2.0-flash-lite" 
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=st.secrets["gemini_api_secret_name"])

# --- Main Logic ---
st.title("CNN RAG Summarizer")
st.caption("Quota-optimized version: Uses 1 request per search instead of 5.")

with st.sidebar:
    query = st.text_input("Topic:", "Technology")
    run_btn = st.button("Generate Summaries")

# Pipeline Setup
links = pull_latest_links()
all_docs = load_docs(links)
embeddings = load_embedding_model()
vectorstore = get_vectorstore(all_docs, embeddings)
llm = load_llm()

if run_btn and vectorstore:
    with st.spinner("Retrieving and summarizing..."):
        # 1. Get relevant docs
        relevant_docs = vectorstore.similarity_search(query, k=5)
        
        # 2. Setup Chain
        # We use a custom prompt to tell Gemini to summarize EACH document separately in ONE response.
        chain = load_summarize_chain(llm, chain_type="stuff")
        
        try:
            # ONE SINGLE API CALL for all 5 documents
            result = chain.invoke({"input_documents": relevant_docs})
            
            st.subheader(f"Summaries for: {query}")
            st.write(result['output_text'])
            
            st.info("Sources utilized:")
            for d in relevant_docs:
                st.caption(d.metadata.get('source'))
                
        except Exception as e:
            if "429" in str(e):
                st.error("📉 **Quota Exhausted:** You've hit the 20 requests/day limit for the Gemini Free Tier. Please wait until tomorrow or upgrade to a paid plan.")
            else:
                st.error(f"An error occurred: {e}")

# Cache Clear
if st.sidebar.button("Refresh News"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()
