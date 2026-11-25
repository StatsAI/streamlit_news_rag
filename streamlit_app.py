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
import numpy as np

# --- Page Config & Styling ---
st.set_page_config(page_title="CNN News Intelligence", layout="wide")

try:
    import base64
    with open('images/picture.png', 'rb') as f:
        logo_b64 = base64.b64encode(f.read()).decode()
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
        st.markdown(
            f'<div style="text-align: center; margin-top: -25px;">'
            f'<img src="data:image/png;base64,{logo_b64}" style="width: 100%;">'
            f'</div>',
            unsafe_allow_html=True
        )
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

        return list(set(links)) # Removed [:20] to ensure no articles are missed
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
            full_text = "\n".join([p for p in paragraphs if len(p) > 20]) # Lowered threshold to avoid missing content
            
            if len(full_text) < 100: # Lowered threshold
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
    else:           
        if groq:
            try:
                chain = load_summarize_chain(groq, chain_type="stuff")
                res = chain.invoke({"input_documents": relevant_docs})
                return res['output_text'], "Groq (Llama 3.3)"
            except Exception as e:
                return f"Error: All models failed. ({e})", "None"
        
        return "Error: No models available. Check your API keys.", "None"

# --- Rundown Logic ---

RUNDOWN_TOPICS = 5

def _cluster_embeddings(vectors, k, iters=15):
    """Small cosine k-means in numpy (deterministic, runs in milliseconds)."""
    X = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
    # Farthest-point initialization so clusters start spread out
    centers = [X[0]]
    for _ in range(1, k):
        sims = np.max(X @ np.array(centers).T, axis=1)
        centers.append(X[int(np.argmin(sims))])
    centers = np.array(centers)
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        labels = np.argmax(X @ centers.T, axis=1)
        for c in range(k):
            members = X[labels == c]
            if len(members):
                center = members.mean(axis=0)
                centers[c] = center / (np.linalg.norm(center) + 1e-10)
    return labels

def _summarize_topic(excerpts):
    prompt = (
        "Below are excerpts from related CNN news articles.\n"
        "Reply in exactly this format:\n"
        "TOPIC: <major topic, 5 words or less>\n"
        "SUMMARY: <2-3 sentence brief of the key developments>\n\n"
        + "\n\n---\n\n".join(excerpts)
    )
    for get_llm, name in ((get_openai, "OpenAI (GPT-4o-Mini)"), (get_groq_fallback, "Groq (Llama 3.3)")):
        llm = get_llm()
        if not llm:
            continue
        try:
            text = llm.bind(max_tokens=200).invoke(prompt).content
            topic, summary = "Top Stories", text.strip()
            for line in text.splitlines():
                if line.upper().startswith("TOPIC:"):
                    topic = line.split(":", 1)[1].strip().replace(".", "")
                elif line.upper().startswith("SUMMARY:"):
                    summary = line.split(":", 1)[1].strip()
            return topic, summary, name
        except Exception:
            continue
    return "Top Stories", "Error: No models available. Check your API keys.", "None"

@st.cache_data(ttl="1d", show_spinner=False)
def build_rundown(source_urls, _vectorstore, _docs):
    # Reuse the chunk embeddings already in Chroma instead of re-embedding
    data = _vectorstore.get(include=["embeddings", "metadatas"])
    by_url = {}
    for emb, meta in zip(data["embeddings"], data["metadatas"]):
        by_url.setdefault(meta.get("source"), []).append(emb)

    text_by_url = {d.metadata["source"]: d.page_content for d in _docs}
    urls = [u for u in by_url if u in text_by_url]
    if not urls:
        return []

    article_vectors = np.array([np.mean(by_url[u], axis=0) for u in urls])
    k = min(RUNDOWN_TOPICS, len(urls))
    labels = _cluster_embeddings(article_vectors, k)

    groups = [[urls[i] for i in range(len(urls)) if labels[i] == c] for c in range(k)]
    groups = sorted([g for g in groups if g], key=len, reverse=True)  # biggest stories first

    def process_group(group_urls):
        excerpts = [text_by_url[u][:400] for u in group_urls[:8]]
        topic, summary, model_name = _summarize_topic(excerpts)
        return {"topic": topic, "summary": summary, "model": model_name, "sources": group_urls}

    with ThreadPoolExecutor(max_workers=k) as executor:
        return list(executor.map(process_group, groups))

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
    rundown_button = st.button('Get Rundown')
    
    st.markdown("---")
    if st.sidebar.button('Clear Cache & Refresh'):
        pull_latest_links.clear()
        load_docs_parallel.clear()
        load_vector_database.clear()
        build_rundown.clear()
        st.cache_resource.clear()
        st.rerun()

# Execution Logic
if (run_button or (query and query != st.session_state.get('last_query', ""))) and vectorstore and not rundown_button:
    st.session_state['last_query'] = query
    
    with st.spinner(f"Analyzing articles for '{query}'..."):
        # Increased k to 15 to ensure we find articles even if vector search hits multiple chunks of same doc
        relevant_docs = vectorstore.similarity_search(query, k=15)
        
        if not relevant_docs:
            st.warning("No relevant articles found.")
        else:
            seen_urls = set()
            unique_docs = []
            
            # Filter duplicates based on source URL
            for doc in relevant_docs:
                url = doc.metadata.get('source')
                if url not in seen_urls:
                    seen_urls.add(url)
                    unique_docs.append(doc)
            
            # Limit to the top 5 unique articles for display
            unique_docs = unique_docs[:5]

            def process_doc(doc):
                topic = get_article_topic(doc)
                summary_text, model_name = run_hybrid_summarization([doc])
                source_url = doc.metadata.get('source', 'CNN Lite')
                return topic, summary_text, model_name, source_url

            with ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(process_doc, unique_docs))
            
            for topic, summary_text, model_name, source_url in results:
                st.markdown(f"### Summary: {topic}")
                st.write(summary_text)
                st.write(f"**Source:** {source_url}")
                st.divider()

            st.caption(f"Generated via {model_name}")

# Rundown Logic
if rundown_button and vectorstore:
    start = time.perf_counter()
    with st.spinner("Building your news rundown..."):
        rundown = build_rundown(tuple(sorted(d.metadata["source"] for d in docs)), vectorstore, docs)
    elapsed = time.perf_counter() - start

    st.header("Your News Rundown")
    if not rundown:
        st.warning("No articles available for a rundown.")
    for item in rundown:
        st.markdown(f"### {item['topic']}")
        st.write(item["summary"])
        st.write("**Sources:**")
        for url in item["sources"]:
            st.markdown(f"- {url}")
        st.divider()

    if rundown:
        st.caption(f"Generated via {rundown[0]['model']} in {elapsed:.1f}s")
