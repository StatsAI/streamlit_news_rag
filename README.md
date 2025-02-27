# Try the app! https://appnewsrag-r3csvfshsyrhujnxef9hzj.streamlit.app/

# What it does: CNN Article Summarization via LangChain, RAG, and Gemini

# How it works: 
## 1. This app ingests the latest articles from cnn into a chromadb vector database using the unstructured library 
## 2. The user's query retrieves the 5 most relevant articles from the vector database 
## 3. These results are passed to an LLM as context for summarization












