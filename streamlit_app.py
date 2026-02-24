__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3
import time
import streamlit as st
#from unstructured.partition.html import partition_html
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
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

# @st.cache_data(ttl="1d")
# def pull_latest_links():
#     cnn_lite_url = "https://lite.cnn.com/"
#     try:
#         elements = partition_html(url=cnn_lite_url)
#         links = []
#         for element in elements:
#             if element.metadata.link_urls:
#                 rel = element.metadata.link_urls[0]
#                 full_url = rel if rel.startswith('http') else f"{cnn_lite_url}{rel}"
#                 links.append(full_url)
#         return links[1:-2] if len(links) > 2 else links
#     except Exception as e:
#         st.error(f"Scraping Error: {e}")
#         return []



@st.cache_resource(ttl="1d")
def load_vector_database(_embedding_func, _docs):
    if not _docs:
        return None

    # 🔹 ADD TEXT SPLITTING
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

# @st.cache_resource(ttl="1d")
# def load_vector_database(_embedding_func, _docs):
#     if not _docs: return None
#     client = chromadb.PersistentClient(path=".chromadb")
#     try:
#         client.delete_collection("cnn_docs")
#     except:
#         pass 
#     return Chroma.from_documents(documents=_docs, embedding=_embedding_func, 
#                                  collection_name="cnn_docs", client=client)


@st.cache_resource(ttl="1d")
def load_docs_parallel(urls):
    docs = []

    def fetch(url):
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
    
            # Better filtering
            article = soup.find("div", class_="article__content") or soup
    
            paragraphs = article.find_all("p")
            text = " ".join(
                p.get_text().strip()
                for p in paragraphs
                if len(p.get_text().strip()) > 50
            )
    
            if len(text) < 500:
                return None  # Skip bad pages
    
            return Document(
                page_content=text,
                metadata={"source": url}
            )
    
        except:
            return None

    # def fetch(url):
    #     try:
    #         response = requests.get(url, timeout=10)
    #         soup = BeautifulSoup(response.text, "html.parser")

    #         paragraphs = soup.find_all("p")
    #         text = " ".join([p.get_text() for p in paragraphs])

    #         return Document(
    #             page_content=text,
    #             metadata={"source": url}
    #         )
    #     except:
    #         return None

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch, urls))

    return [doc for doc in results if doc]

# @st.cache_resource(ttl="1d")
# def load_docs_parallel(urls):
#     with ThreadPoolExecutor() as executor:
#         loaders = [UnstructuredURLLoader(urls=[u]) for u in urls]
#         docs_list = list(executor.map(lambda l: l.load(), loaders))
#     return [doc for sublist in docs_list for doc in sublist]

# --- Hybrid LLM Logic (Gemini -> Groq Fallback) ---

@st.cache_resource
def get_openai():
    try:
        return ChatOpenAI(
            model="gpt-5-mini",
            api_key = st.secrets.get("open_ai_api_key", ""),
            temperature=0
        )
    except: return None
        
@st.cache_resource
def get_gemini():
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=st.secrets["gemini_api_secret_name"],
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
    """Helper function to extract a short topic from the article text."""
    #llm = get_groq_fallback() # Using Groq for quick extraction
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
            return res['output_text'], "OpenAI (GPT-5-Mini)"
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
st.info("Status: Primary (OpenAI GPT-5-Mini) | Fallback (Groq Llama 3.3)")

status_ui = st.empty()
status_ui.info("Fetching latest news...")
links = pull_latest_links()
docs = load_docs_parallel(links)
vectorstore = load_vector_database(load_embedding_model(), docs)
status_ui.success("System Ready!")

# Sidebar Input
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
# --- Execution Logic (Updated for Parallel Processing) ---

if (run_button or (query and query != st.session_state.get('last_query', ""))) and vectorstore:
    st.session_state['last_query'] = query
    
    with st.spinner(f"Analyzing articles for '{query}'..."):
        relevant_docs = vectorstore.similarity_search(query, k=5)
        
        if not relevant_docs:
            st.warning("No relevant articles found.")
        else:
            # Define a helper to process a single document for parallel execution
            def process_doc(doc):
                topic = get_article_topic(doc)
                summary_text, model_name = run_hybrid_summarization([doc])
                source_url = doc.metadata.get('source', 'CNN Lite')
                return topic, summary_text, model_name, source_url

            # Execute summarization in parallel
            with ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(process_doc, relevant_docs))

            # Display results
            for topic, summary_text, model_name, source_url in results:
                st.markdown(f"### Summary: {topic}")
                st.write(summary_text)
                st.write(f"**Source:** {source_url}")
                st.divider()

            st.caption(f"Generated via {model_name}")


# if (run_button or (query and query != st.session_state.get('last_query', ""))) and vectorstore:
#     st.session_state['last_query'] = query
    
#     with st.spinner(f"Analyzing articles for '{query}'..."):
#         relevant_docs = vectorstore.similarity_search(query, k=5)
        
#         if not relevant_docs:
#             st.warning("No relevant articles found.")
#         else:
#             for doc in relevant_docs:
#                 # Step 1: Extract Topic
#                 topic = get_article_topic(doc)
                
#                 # Step 2: Generate Summary
#                 summary_text, model_name = run_hybrid_summarization([doc])
#                 source_url = doc.metadata.get('source', 'CNN Lite')
                
#                 # Step 3: Display with requested format
#                 st.markdown(f"### Summary: {topic}")
#                 st.write(summary_text)
#                 st.write(f"**Source:** {source_url}")
#                 st.divider()

#             st.caption(f"Generated via {model_name}")
