__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3

import streamlit as st
from unstructured.partition.html import partition_html
from langchain.document_loaders import UnstructuredURLLoader
import chromadb
from chromadb.config import Settings
from langchain.vectorstores.chroma import Chroma
#from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
#from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
#import asyncio

# Ensure the event loop is set correctly
# try:
#     loop = asyncio.get_event_loop()
# except RuntimeError as e:
#     if str(e).startswith('There is no current event loop in thread'):
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#     else:
#         raise

# Streamlit app title
st.title("CNN Article Summarizer")

# Cache the embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Cache the links
@st.cache_data
def pull_latest_links():
    cnn_lite_url = "https://lite.cnn.com/"
    elements = partition_html(url=cnn_lite_url)
    links = []

    for element in elements:
        try:
            if element.metadata.link_urls:
                relative_link = element.metadata.link_urls[0]
                links.append(f"{cnn_lite_url}{relative_link}")
        except IndexError:
            continue

    links = links[1:-2]
    return links

# Cache the ChromaDB client
@st.cache_resource
def get_chroma_client():
    return chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=".chromadb"  # Persist data to this directory
    ))

# Cache the vector database
@st.cache_resource
def load_vector_database(_embedding_function, _docs):
    chroma_client = get_chroma_client()
    
    # Delete existing collection if it exists
    try:
        chroma_client.delete_collection("cnn_doc_embeddings")
    except ValueError:
        pass  # Collection does not exist

    return Chroma.from_documents(_docs, _embedding_function, collection_name="cnn_doc_embeddings", client=chroma_client)

# Button to pull the latest links
if st.button("Pull Latest Links"):
    links = pull_latest_links()
    st.session_state['links'] = links
    st.write("Latest links pulled successfully!")

# Textbox for user query
query = st.text_input("Enter your query:")

if 'links' in st.session_state and query:
    links = st.session_state['links']
    
    # Load documents
    loaders = UnstructuredURLLoader(urls=links, show_progress_bar=True)
    docs = loaders.load()

    # Load embedding model
    embedding_function = load_embedding_model()

    # Load vector database
    vectorstore = load_vector_database(embedding_function, docs)

    # Query the vector database
    query_docs = vectorstore.similarity_search(query, k=5)

    # Load Gemini model
    gem_api_key = st.secrets["gemini_api_secret_name"]
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                 temperature=0.7,
                                 max_tokens=None,
                                 timeout=None,
                                 max_retries=2,
                                 google_api_key=gem_api_key)

    # Summarize the results
    chain = load_summarize_chain(llm, chain_type="stuff")

    for doc in query_docs:
        source = doc.metadata
        result = chain.invoke([doc])
        st.write(result['output_text'])
        st.write(source)
        st.write('')
