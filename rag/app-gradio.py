# Import Gradio for building the web app UI
import gradio as gr
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

# Cache Wikipedia docs and their embeddings/vectorstore for each question
from functools import lru_cache

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

def rag_answer(question):
    wiki_docs, vectorstore = get_wikipedia_docs_and_vectorstore(question)
    if not wiki_docs or vectorstore is None:
        return "No relevant Wikipedia content found.", []
    retriever = vectorstore.as_retriever()
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Use the following context to answer the question.\nContext: {context}\nQuestion: {question}\nAnswer:"
    )
    llm = OllamaLLM(model="llama2")
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    relevant_docs = retriever.get_relevant_documents(question)
    answer = rag_chain.invoke(question)
    return answer, relevant_docs

def llm_only_answer(question):
    llm = OllamaLLM(model="llama2")
    return llm.invoke(question)

def gradio_interface(question):
    start_time = time.time()
    rag_ans, _ = rag_answer(question)
    llm_ans = llm_only_answer(question)
    elapsed = time.time() - start_time
    return (
        f"⏱️ Query time: {elapsed:.2f} seconds",
        rag_ans if rag_ans else "None",
        llm_ans if llm_ans else "None"
    )

def rag_answer_stream(question, progress=gr.Progress(track_tqdm=True)):
    wiki_docs, vectorstore = get_wikipedia_docs_and_vectorstore(question)
    if not wiki_docs or vectorstore is None:
        yield f"⏱️ Query time: 0.00 seconds", [["RAG", "No relevant Wikipedia content found."]], [["LLM", "None"]]
        return
    retriever = vectorstore.as_retriever()
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Use the following context to answer the question.\nContext: {context}\nQuestion: {question}\nAnswer:"
    )
    llm = OllamaLLM(model="llama2", streaming=True)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = ""
    start_time = time.time()
    for chunk in rag_chain.stream(question):
        answer += chunk
        elapsed = time.time() - start_time
        # Only update the last message in the chatbot, not add a new one each time
        yield f"⏱️ Query time: {elapsed:.2f} seconds", [["RAG", answer]], [["LLM", "..."]]

def llm_only_answer_stream(question):
    llm = OllamaLLM(model="llama2", streaming=True)
    answer = ""
    start_time = time.time()
    for chunk in llm.stream(question):
        answer += chunk
        elapsed = time.time() - start_time
        yield [["LLM", answer]]

def gradio_interface_stream(question, progress=gr.Progress(track_tqdm=True)):
    rag_stream = rag_answer_stream(question, progress)
    rag_time_msg, rag_history, llm_history = next(rag_stream)
    # Start LLM stream in parallel
    llm_stream = llm_only_answer_stream(question)
    for rag_time_msg, rag_history, _ in rag_stream:
        yield rag_time_msg, rag_history, llm_history
    for llm_history in llm_stream:
        yield rag_time_msg, rag_history, llm_history

demo = gr.Interface(
    fn=gradio_interface_stream,
    inputs=gr.Textbox(label="Enter your question:"),
    outputs=[
        gr.Textbox(label="Query Time"),
        gr.Chatbot(label="RAG Answer (Wikipedia-Augmented)"),
        gr.Chatbot(label="LLM-Only Answer (No Wikipedia)")
    ],
    title="LangChain RAG with Ollama Demo",
    description="This is a simple Gradio app demonstrating Retrieval-Augmented Generation (RAG) using LangChain and Ollama.",
    allow_flagging="never",
    live=False,
    concurrency_limit=1
)

demo.queue().launch(share=True)
