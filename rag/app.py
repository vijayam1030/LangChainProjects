# Import Streamlit for building the web app UI
#streamlit run rag/app.py --server.port 8502
import streamlit as st
# Import Ollama LLM and Embeddings from langchain_ollama (for LLM and vector search)
from langchain_ollama import OllamaLLM, OllamaEmbeddings
# Import PromptTemplate for formatting prompts to the LLM
from langchain.prompts import PromptTemplate
# Import FAISS for vector database (retrieval)
from langchain_community.vectorstores import FAISS
# Import CharacterTextSplitter for splitting documents into chunks
from langchain.text_splitter import CharacterTextSplitter
# Import RunnablePassthrough for chaining steps in the pipeline
from langchain.schema.runnable import RunnablePassthrough
# Import StrOutputParser to parse LLM output as string
from langchain.schema.output_parser import StrOutputParser
# Import WikipediaLoader to fetch Wikipedia articles
from langchain_community.document_loaders import WikipediaLoader
import time

# Set up Streamlit page configuration (title and layout)
st.set_page_config(page_title="LangChain RAG Demo", layout="wide", initial_sidebar_state="collapsed")
# Remove custom CSS for dark mode and improved UI
# Set the main title of the app
st.title("LangChain RAG with Ollama Demo")
# Write a description for the app
st.write("This is a simple Streamlit app demonstrating Retrieval-Augmented Generation (RAG) using LangChain and Olloma.")

# Create a text input box for the user to enter their question
question = st.text_input("Enter your question:")

# Cache Wikipedia document retrieval to avoid repeated downloads for the same query
@st.cache_data(show_spinner=False)
def get_wikipedia_docs(query, lang="en", max_docs=5):
    # Create a WikipediaLoader to fetch articles
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=max_docs)
    # Load and return the documents
    return loader.load()

# Cache Wikipedia docs and their embeddings/vectorstore for each question
@st.cache_resource(show_spinner=False)
def get_wikipedia_docs_and_vectorstore(question, lang="en", max_docs=5):
    loader = WikipediaLoader(query=question, lang=lang, load_max_docs=max_docs)
    wiki_docs = loader.load()
    docs = [doc.page_content for doc in wiki_docs]
    if not docs:
        return [], None
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    split_docs = text_splitter.create_documents(docs)
    embeddings = OllamaEmbeddings(model="llama2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return wiki_docs, vectorstore

# Main function to run the RAG pipeline and return an answer
def rag_answer(question):
    # 1. Retrieve relevant Wikipedia documents and vectorstore for the question
    wiki_docs, vectorstore = get_wikipedia_docs_and_vectorstore(question)
    if not wiki_docs or vectorstore is None:
        return "No relevant Wikipedia content found.", []
    retriever = vectorstore.as_retriever()
    # 4. Create a prompt template for the LLM
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Use the following context to answer the question.\nContext: {context}\nQuestion: {question}\nAnswer:"
    )
    # Initialize the Ollama LLM
    llm = OllamaLLM(model="llama2")
    # 5. Build the RAG chain: retrieve, prompt, generate, and parse output
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    # 5. Retrieve the most relevant chunks for the question
    relevant_docs = retriever.get_relevant_documents(question)
    # 6. Run the chain with the user's question and return the answer and sources
    answer = rag_chain.invoke(question)
    return answer, relevant_docs

# Function to get the LLM-only answer
def llm_only_answer(question):
    # Directly use the LLM without Wikipedia context
    llm = OllamaLLM(model="llama2")
    return llm.invoke(question)

# If the user has entered a question
if question:
    # Display the user's question
    st.write(f"You asked: {question}")
    start_time = time.time()
    # RAG answer
    rag_ans, _ = rag_answer(question)
    # LLM-only answer
    llm_ans = llm_only_answer(question)
    elapsed = time.time() - start_time
    # Display query time at the top right
    st.markdown(f"<div style='position: absolute; top: 10px; right: 30px; font-size: 18px; color: #e0e0e0;'>⏱️ Query time: {elapsed:.2f} seconds</div>", unsafe_allow_html=True)
    # Display RAG answer section
    st.subheader("RAG Answer (Wikipedia-Augmented)")
    st.info(rag_ans if rag_ans else "None")
    # Display LLM-only answer section
    st.subheader("LLM-Only Answer (No Wikipedia)")
    st.info(llm_ans if llm_ans else "None")
