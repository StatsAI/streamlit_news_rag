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
from PIL import ImageOps

#logo = Image.open('images/picture.png')
#st.image(logo)
#newsize = (95, 95)
#logo = logo.resize(newsize)

# # Centering the logo
# col1, col2, col3 = st.columns([1, 2, 1])  # Create columns with relative widths
# with col2:  # Place the logo in the middle column
#     st.image(logo)

logo = Image.open('images/picture.png')
#newsize = (95, 95)
#logo = logo.resize(newsize)

st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: 30px;
            margin-right: auto;
	    margin-top: -25px;
            width: 100%;
	    #margin: 0;	         		
        }
    </style>
    """, unsafe_allow_html=True
)

with st.sidebar:
    st.image(logo)

st.markdown("""
        <style>
               .block-container {
		    padding-top: 0;
                }
        </style>
        """, unsafe_allow_html=True)

#st.sidebar.write('')
#st.sidebar.write('')

with st.sidebar:
    st.markdown("""
        <style>
            [data-testid=stTextInput] {
                height: -5px;  # Adjust the height as needed
            }
        </style>
    """, unsafe_allow_html=True)

    query = st.text_input("Topic Selection: Enter the topic you want to summarize articles for", "Trump", key = "text")

# Streamlit app title
st.title("CNN Article Summarization via LangChain, RAG, and Gemini")
st.write("How this app works: This app ingests the latest articles from cnn into a chromadb vector database using the unstructured library. The user's query retrieves the 5 most relevant articles from the vector database. These results are passed to an LLM as context for summarization")
st.write('<a href="https://lite.cnn.com/">Click here to visit CNN Lite</a>', unsafe_allow_html=True)

# Cache the embedding model
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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
    return chromadb.PersistentClient(path=".chromadb")  # Use PersistentClient and path

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

@st.cache_resource
# Load documents in parallel
def load_documents_parallel(urls):
    with ThreadPoolExecutor() as executor:
        loaders = [UnstructuredURLLoader(urls=[url], show_progress_bar=False) for url in urls]
        docs = list(executor.map(lambda loader: loader.load(), loaders))
    return [doc for sublist in docs for doc in sublist]  # Flatten the list

@st.cache_resource
# Load gemini model
def load_gemini_model():    
    gem_api_key = st.secrets["gemini_api_secret_name"]
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                 temperature=0.7,
                                 max_tokens=None,
                                 timeout=None,
                                 max_retries=2,
                                 google_api_key=gem_api_key)
    return llm

start = time.time()
links = pull_latest_links()
st.session_state['links'] = links
end = time.time()
diff = end - start
#st.sidebar.write(f"Latest links pulled in {round(diff,3)} seconds")

start = time.time()
docs = load_documents_parallel(links)
end = time.time()
diff = end - start
#st.sidebar.write(f"Content extracted from links in {round(diff,3)} seconds")

start = time.time()
embedding_function = load_embedding_model()
end = time.time()
diff = end - start
#st.sidebar.write(f"Embedding model loaded in {round(diff,3)} seconds")

#st.write("Embedding model loaded!")
start = time.time()
vectorstore = load_vector_database(embedding_function, docs)
end = time.time()
diff = end - start
#st.sidebar.write(f"Vector database loaded in {round(diff,3)} seconds")

start = time.time()
llm = load_gemini_model()
end = time.time()
diff = end - start
#st.sidebar.write(f"Gemini model loaded in {round(diff,3)} seconds")

# Textbox for user query
#query = st.text_input("Enter your query:")

# if st.sidebar.button('Get latest links'):
# 	start = time.time()
# 	links = pull_latest_links()
# 	st.session_state['links'] = links
# 	end = time.time()
# 	diff = end - start
# 	#st.sidebar.write(f"Latest links pulled in {round(diff,3)} seconds")
	
# 	start = time.time()
# 	docs = load_documents_parallel(links)
# 	end = time.time()
# 	diff = end - start
# 	#st.sidebar.write(f"Content extracted from links in {round(diff,3)} seconds")
	
# 	# start = time.time()
# 	# embedding_function = load_embedding_model()
# 	# end = time.time()
# 	# diff = end - start
# 	# st.write(f"Embedding model loaded in {round(diff,3)} seconds")
	
# 	#st.write("Embedding model loaded!")
# 	start = time.time()
# 	vectorstore = load_vector_database(embedding_function, docs)
# 	end = time.time()
# 	diff = end - start
# 	st.sidebar.write(f"Vector database loaded in {round(diff,3)} seconds")
	
# 	# start = time.time()
# 	# llm = load_gemini_model()
# 	# end = time.time()
# 	# diff = end - start
# 	# st.write(f"Gemini model loaded in {round(diff,3)} seconds")

if query:    
    # Query the vector database
    start = time.time()
    query_docs = vectorstore.similarity_search(query, k=5)
    end = time.time()
    #st.sidebar.write(f"Vector database queried in {round(diff,3)} seconds")    

    # Summarize the results
    start = time.time()         
    chain = load_summarize_chain(llm, chain_type="stuff")
    end = time.time()
    #st.sidebar.write(f"Chain summarized in {round(diff,3)} seconds") 

    # Display results
    for doc in query_docs:
        source = doc.metadata
        result = chain.invoke([doc])
	st.write(result['output_text'])
	string = '<a href=' + source + '</a>'
	st.write(string, unsafe_allow_html=True)
        st.write(source)
        st.write('')
