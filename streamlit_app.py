__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3
import time
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_classic.prompts import PromptTemplate
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
    st.sidebar.warning("Logo not found. Please check 'images/picture.png' path.")

# --- Backend Functions ---

@st.cache_resource()
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_data(ttl="1d")
def pull_latest_links():
    cnn_lite_url = "https://lite.cnn.com/"
    try:
        response = requests.get(cnn_lite_url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/202" in href:  # CNN article pattern
                full_url = href if href.startswith("http") else f"{cnn_lite_url}{href}"
                links.append(full_url)

        return list(set(links))[:20]
    except Exception as e:
        st.error(f"Scraping Error: {e}")
        return []

@st.cache_resource(ttl="1d")
def load_vector_database(_embedding_func, _docs):
    if not _docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    split_docs = text_splitter.split_documents(_docs)

    client = chromadb.PersistentClient(path=".chromadb")

    try:
        client.delete_collection("cnn_docs")
    except:
        pass

    return Chroma.from_documents(
        documents=split_docs,
        embedding=_embedding_func,
        collection_name="cnn_docs",
        client=client
    )

@st.cache_resource(ttl="1d")
def load_docs_parallel(urls):
    def fetch(url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, timeout=10, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.extract()
            
            paragraphs = [p.get_text().strip() for p in soup.find_all("p")]
            full_text = "\n".join([p for p in paragraphs if len(p) > 50])
            
            if len(full_text) < 400 or ("updated" in full_text.lower() and len(full_text) < 600):
                return None
    
            return Document(
                page_content=full_text,
                metadata={"source": url}
            )
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(fetch, urls))

    return [doc for doc in results if doc is not None]

# --- Hybrid LLM Logic ---

@st.cache_resource
def get_openai():
    try:
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key = st.secrets.get("open_ai_api_key", ""),
            temperature=0
        )
    except: return None
        
@st.cache_resource
def get_gemini():
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=st.secrets["gemini_api_key"],
            temperature=0.7
        )
    except: return None

@st.cache_resource
def get_groq_fallback():
    try:
        return ChatGroq(
            model_name="llama-3.3-70b-versatile",
            groq_api_key=st.secrets["groq_api_key"],
            temperature=0.7
        )
    except: return None

def get_article_topic(doc):
    llm = get_openai()
    if not llm: return "Article"
    
    prompt = f"Identify the main topic of this article in 5 words or less: {doc.page_content[:1000]}"
    try:
        response = llm.invoke(prompt)
        return response.content.strip().replace(".", "")
    except:
        return "Article Content"

def run_hybrid_summarization(relevant_docs):
    groq = get_groq_fallback()
    open_ai = get_openai()
    
    if open_ai:
        try:
            chain = load_summarize_chain(open_ai, chain_type="stuff")
            res = chain.invoke({"input_documents": relevant_docs})
            return res['output_text'], "OpenAI (GPT-4o-Mini)"
        except Exception as e:
            return f"Error: All models failed. ({e})", "None"
    
    if groq:
        try:
            chain = load_summarize_chain(groq, chain_type="stuff")
            res = chain.invoke({"input_documents": relevant_docs})
            return res['output_text'], "Groq (Llama 3.3)"
        except Exception as e:
            return f"Error: All models failed. ({e})", "None"
    
    return "Error: No models available. Check your API keys.", "None"

# --- UI Layout ---

st.title("CNN RAG Intelligence")
st.info("Status: Primary (OpenAI) | Fallback (Groq)")

status_ui = st.empty()
status_ui.info("Fetching latest news...")
links = pull_latest_links()
docs = load_docs_parallel(links)
vectorstore = load_vector_database(load_embedding_model(), docs)
status_ui.success("System Ready!")

with st.sidebar:
    query = st.text_input("Search Topic:", value="Trump")
    run_button = st.button('Generate Summaries')
    
    st.markdown("---")
    if st.sidebar.button('Clear Cache & Refresh'):
        pull_latest_links.clear()
        load_docs_parallel.clear()
        load_vector_database.clear()
        st.cache_resource.clear()
        st.rerun()

# Execution Logic
if (run_button or (query and query != st.session_state.get('last_query', ""))) and vectorstore:
    st.session_state['last_query'] = query
    
    with st.spinner(f"Analyzing articles for '{query}'..."):
        relevant_docs = vectorstore.similarity_search(query, k=5)
        
        if not relevant_docs:
            st.warning("No relevant articles found.")
        else:
            def process_doc(doc):
                topic = get_article_topic(doc)
                summary_text, model_name = run_hybrid_summarization([doc])
                source_url = doc.metadata.get('source', 'CNN Lite')
                return topic, summary_text, model_name, source_url

            with ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(process_doc, relevant_docs))
            
            for topic, summary_text, model_name, source_url in results:
                st.markdown(f"### Summary: {topic}")
                st.write(summary_text)
                st.write(f"**Source:** {source_url}")
                st.divider()

            st.caption(f"Generated via {model_name}")
