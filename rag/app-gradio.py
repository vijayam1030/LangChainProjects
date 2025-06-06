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

# List of available LLM models
LLM_MODELS = ["llama2", "mistral", "phi3", "qwen3:1.7b", "gemma3:1b", "deepseek-r1:1.5b"]

def get_wikipedia_docs_and_vectorstore(question, lang="en", max_docs=5):
    loader = WikipediaLoader(query=question, lang=lang, load_max_docs=max_docs)
    wiki_docs = loader.load()
    docs = [doc.page_content for doc in wiki_docs]
    if not docs:
        return [], None
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    split_docs = text_splitter.create_documents(docs)
    embeddings = OllamaEmbeddings(model="llama2")  # Always use Wikipedia for RAG
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return wiki_docs, vectorstore

def rag_answer(question, llm_model):
    wiki_docs, vectorstore = get_wikipedia_docs_and_vectorstore(question)
    if not wiki_docs or vectorstore is None:
        return "No relevant Wikipedia content found.", []
    retriever = vectorstore.as_retriever()
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Use the following context to answer the question.\nContext: {context}\nQuestion: {question}\nAnswer:"
    )
    llm = OllamaLLM(model=llm_model)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    relevant_docs = retriever.get_relevant_documents(question)
    answer = rag_chain.invoke(question)
    return answer, relevant_docs

def llm_only_answer(question, llm_model):
    llm = OllamaLLM(model=llm_model)
    return llm.invoke(question)

def gradio_interface(question, llm_model):
    start_time = time.time()
    rag_ans, _ = rag_answer(question, llm_model)
    llm_ans = llm_only_answer(question, llm_model)
    elapsed = time.time() - start_time
    return (
        f"⏱️ Query time: {elapsed:.2f} seconds",
        rag_ans if rag_ans else "None",
        llm_ans if llm_ans else "None"
    )

def rag_answer_stream(question, llm_model, progress=gr.Progress(track_tqdm=True)):
    wiki_docs, vectorstore = get_wikipedia_docs_and_vectorstore(question)
    if not wiki_docs or vectorstore is None:
        yield f"⏱️ Query time: 0.00 seconds", [["RAG", "No relevant Wikipedia content found."]], [["LLM", "None"]]
        return
    retriever = vectorstore.as_retriever()
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Use the following context to answer the question.\nContext: {context}\nQuestion: {question}\nAnswer:"
    )
    llm = OllamaLLM(model=llm_model, streaming=True)
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
        # Always yield both RAG and LLM-only answers, updating the RAG chatbot
        yield f"⏱️ Query time: {elapsed:.2f} seconds", [["RAG", answer]], [["LLM", "Waiting..."]]
    # After RAG is done, yield final RAG and let LLM-only stream start

def llm_only_answer_stream(question, llm_model):
    llm = OllamaLLM(model=llm_model, streaming=True)
    answer = ""
    start_time = time.time()
    for chunk in llm.stream(question):
        answer += chunk
        elapsed = time.time() - start_time
        yield [["LLM", answer]]

def gradio_interface_stream(question, llm_model, progress=gr.Progress(track_tqdm=True)):
    rag_stream = rag_answer_stream(question, llm_model, progress)
    rag_time_msg, rag_history, llm_history = next(rag_stream)
    yield rag_time_msg, rag_history, llm_history
    for rag_time_msg, rag_history, _ in rag_stream:
        yield rag_time_msg, rag_history, llm_history
    # Now start LLM-only stream
    llm_stream = llm_only_answer_stream(question, llm_model)
    llm_final = None
    for llm_history in llm_stream:
        llm_final = llm_history
        yield rag_time_msg, rag_history, llm_history
    # After LLM-only is done, yield both final answers one last time (ensures both are visible)
    if llm_final is not None:
        yield rag_time_msg, rag_history, llm_final

with gr.Blocks() as demo:
    gr.Markdown("# LangChain RAG with Ollama Demo\nThis is a simple Gradio app demonstrating Retrieval-Augmented Generation (RAG) using LangChain and Ollama.")
    with gr.Row():
        question = gr.Textbox(label="Enter your question:")
    with gr.Row():
        llm_model = gr.Dropdown(choices=LLM_MODELS, value="llama2", label="LLM Model (for both RAG and LLM-only)")
        btn = gr.Button("Submit", variant="primary")
    with gr.Row():
        query_time = gr.Textbox(label="Query Time")
    with gr.Row():
        rag_chat = gr.Chatbot(label="RAG Answer (Wikipedia-Augmented)")
        llm_chat = gr.Chatbot(label="LLM-Only Answer (No Wikipedia)")
    btn.click(
        gradio_interface_stream,
        inputs=[question, llm_model],
        outputs=[query_time, rag_chat, llm_chat]
    )

demo.queue().launch(share=True)
