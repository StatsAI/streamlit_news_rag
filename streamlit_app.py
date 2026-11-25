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
import re

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

MIN_TOPICS, MAX_TOPICS = 6, 10   # how many topics the rundown aims for
MAX_BULLETS = 3                  # bullets per topic (one bullet can cover several articles)
OTHER = "Other Headlines"

def _ask_llm(prompt, max_tokens):
    """Try OpenAI first, then Groq. Returns (text, model_name) or (None, 'None')."""
    for get_llm, name in ((get_openai, "OpenAI (GPT-4o-Mini)"), (get_groq_fallback, "Groq (Llama 3.3)")):
        llm = get_llm()
        if not llm:
            continue
        try:
            return llm.bind(max_tokens=max_tokens).invoke(prompt).content, name
        except Exception:
            continue
    return None, "None"

def _url_parts(url):
    return [p for p in url.split("/") if p and p != "index.html"]

def _headline(doc):
    """Short description of an article: CNN section + URL slug + opening words."""
    parts = _url_parts(doc.metadata["source"])
    section = parts[-2] if len(parts) >= 2 else ""
    slug = parts[-1].replace("-", " ") if parts else ""
    opening = " ".join(doc.page_content.split())[:100]
    return f"({section}) {slug} | {opening}"

def _group_articles(docs):
    """One LLM call sorts every article into a manageable number of broad topics.
    Returns a list of (topic name, [doc indexes])."""
    listing = "\n".join(f"[{i + 1}] {_headline(d)}" for i, d in enumerate(docs))
    prompt = (
        f"Group these news articles into {MIN_TOPICS} to {MAX_TOPICS} broad topics by subject.\n"
        "Rules:\n"
        "- Use broad, intuitive topics, e.g. all articles about Trump or Congress go under "
        "'Trump & US Politics'; all about the war in Ukraine under 'Russia-Ukraine War'.\n"
        f"- Stories that don't fit any topic go in '{OTHER}'.\n"
        "- Every article number must appear exactly once.\n"
        "Reply with one line per topic, nothing else, in this format:\n"
        "Topic Name: 1, 4, 7\n\n"
        + listing
    )
    text, _ = _ask_llm(prompt, max_tokens=800)

    groups, used = {}, set()
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        name, nums = line.rsplit(":", 1)
        name = name.strip(" -*#").strip()
        idxs = [int(n) - 1 for n in re.findall(r"\d+", nums)]
        idxs = [i for i in idxs if 0 <= i < len(docs) and i not in used]
        if name and idxs:
            groups.setdefault(name, []).extend(idxs)
            used.update(idxs)

    if text is None:
        # Model unavailable: fall back to CNN's own sections
        for i, d in enumerate(docs):
            parts = _url_parts(d.metadata["source"])
            section = parts[-2].replace("-", " ").title() if len(parts) >= 2 else OTHER
            groups.setdefault(section, []).append(i)
    else:
        # Anything the model skipped goes to Other Headlines
        skipped = [i for i in range(len(docs)) if i not in used]
        if skipped:
            groups.setdefault(OTHER, []).extend(skipped)

    # Keep the topic count manageable even if the model made too many
    ranked = sorted(((n, ix) for n, ix in groups.items() if n != OTHER), key=lambda g: -len(g[1]))
    other = groups.get(OTHER, [])
    for _, ix in ranked[MAX_TOPICS - 1:]:
        other = other + ix
    result = ranked[:MAX_TOPICS - 1]
    if other:
        result.append((OTHER, other))
    return result

def _summarize_topic(topic, docs):
    excerpt_len = 600 if len(docs) <= 5 else 300
    excerpts = "\n\n".join(
        f"[{i + 1}] {_headline(d)}\n{d.page_content[:excerpt_len]}" for i, d in enumerate(docs[:15])
    )
    if topic == OTHER:
        instructions = (
            "Write exactly one short bullet per article, each one complete sentence "
            "ending with its article number in brackets, e.g. [2]."
        )
    else:
        instructions = (
            f"Then write at most {MAX_BULLETS} bullets. Each bullet is one or two complete sentences "
            "on a key development; combine related articles into the same bullet. End each bullet "
            "with the numbers of every article it draws on, e.g. [1][4]. Cite every article at least once."
        )
    prompt = (
        f"Write a brief for the news topic '{topic}' using the numbered articles below.\n"
        "Reply with only the brief: start with one complete sentence overview of the topic "
        "(skip the overview for 'Other Headlines'), then bullets starting with '- '.\n"
        + instructions + "\n\n" + excerpts
    )
    text, model_name = _ask_llm(prompt, max_tokens=500)
    if not text:
        return "Error: No models available. Check your API keys.", model_name

    # Turn [1], [1][4] or [1, 4] markers into links to the matching articles
    def link(match):
        out = []
        for n in re.findall(r"\d+", match.group(0)):
            i = int(n) - 1
            if 0 <= i < len(docs):
                out.append(f"[[{i + 1}]]({docs[i].metadata['source']})")
        return " ".join(out) if out else match.group(0)

    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    lines = [re.sub(r"^[*•]\s+", "- ", l) for l in lines]  # normalize bullet characters
    overview = [l for l in lines if not l.startswith("- ")]
    bullets = [l for l in lines if l.startswith("- ")]
    if topic != OTHER:
        bullets = bullets[:MAX_BULLETS]
    body = " ".join(overview) + ("\n\n" + "\n".join(bullets) if bullets else "")
    return re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", link, body).strip(), model_name

@st.cache_data(ttl="1d", show_spinner=False)
def build_rundown(source_urls, _docs):
    if not _docs:
        return []

    # Biggest topics first, Other Headlines last
    groups = _group_articles(_docs)

    def process_group(item):
        topic, idxs = item
        topic_docs = [_docs[i] for i in idxs]
        summary, model_name = _summarize_topic(topic, topic_docs)
        return {"topic": topic, "summary": summary, "model": model_name,
                "sources": [d.metadata["source"] for d in topic_docs]}

    # All topics are summarized at the same time
    with ThreadPoolExecutor(max_workers=min(len(groups), 32)) as executor:
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
        rundown = build_rundown(tuple(sorted(d.metadata["source"] for d in docs)), docs)
    elapsed = time.perf_counter() - start

    st.header("Your News Rundown")
    if not rundown:
        st.warning("No articles available for a rundown.")
    for item in rundown:
        st.markdown(f"### {item['topic']}")
        st.markdown(item["summary"])
        st.write("**Sources:**")
        for url in item["sources"]:
            st.markdown(f"- {url}")
        st.divider()

    if rundown:
        st.caption(f"Generated via {rundown[0]['model']} in {elapsed:.1f}s")
