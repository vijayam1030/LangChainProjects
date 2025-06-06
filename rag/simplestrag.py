from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS

# Assuming split_docs is already defined
# Create embeddings and vector store
embeddings = OllamaEmbeddings(model="llama2")
vectorstore = FAISS.from_documents(split_docs, embeddings)