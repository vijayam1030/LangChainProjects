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

# List of available LLM models (only ones that are actually pulled)
LLM_MODELS = ["tinyllama:1.1b"]

def get_wikipedia_docs_and_vectorstore(question, lang="en", max_docs=5):
    loader = WikipediaLoader(query=question, lang=lang, load_max_docs=max_docs)
    wiki_docs = loader.load()
    docs = [doc.page_content for doc in wiki_docs]
    if not docs:
        return [], None
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    split_docs = text_splitter.create_documents(docs)
    embeddings = OllamaEmbeddings(model="tinyllama:1.1b")  # Always use Wikipedia for RAG
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return wiki_docs, vectorstore

# Remove conversation history and context logic, revert to simple stateless answers

def rag_answer(question, llm_model):
    wiki_docs, vectorstore = get_wikipedia_docs_and_vectorstore(question)
    if not wiki_docs or vectorstore is None:
        return "No relevant Wikipedia content found.", []
    retriever = vectorstore.as_retriever()
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Answer the question using ONLY the following context from Wikipedia.\n"
            "If the answer is not in the context, say 'I don't know.'\n"
            "\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
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
        [["LLM", llm_ans]],
        [["RAG", rag_ans]]
    )

def rag_answer_stream(question, llm_model, progress=gr.Progress(track_tqdm=True)):
    wiki_docs, vectorstore = get_wikipedia_docs_and_vectorstore(question)
    if not wiki_docs or vectorstore is None:
        yield f"⏱️ Query time: 0.00 seconds", [["LLM", "None"]], [["RAG", "No relevant Wikipedia content found."]]
        return
    retriever = vectorstore.as_retriever()
    # Concatenate all retrieved docs as context for the prompt
    relevant_docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"Answer the question using ONLY the following context from Wikipedia.\nIf the answer is not in the context, say 'I don't know.'\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    llm = OllamaLLM(model=llm_model, streaming=True)
    answer = ""
    start_time = time.time()
    for chunk in llm.stream(prompt):
        answer += chunk
        elapsed = time.time() - start_time
        yield f"⏱️ Query time: {elapsed:.2f} seconds", [["LLM", "Waiting..."]], [["RAG", answer]]

def llm_only_answer_stream(question, llm_model):
    llm = OllamaLLM(model=llm_model, streaming=True)
    answer = ""
    start_time = time.time()
    for chunk in llm.stream(question):
        answer += chunk
        elapsed = time.time() - start_time
        yield [["LLM", answer]]

def gradio_interface_stream(question, llm_model, progress=gr.Progress(track_tqdm=True)):
    try:
        rag_stream = rag_answer_stream(question, llm_model, progress)
        rag_time_msg, llm_history, rag_history = next(rag_stream)
        yield rag_time_msg, llm_history, rag_history
        for rag_time_msg, llm_history, rag_history in rag_stream:
            yield rag_time_msg, llm_history, rag_history
        llm_stream = llm_only_answer_stream(question, llm_model)
        llm_final = None
        for llm_history in llm_stream:
            llm_final = llm_history
            yield rag_time_msg, llm_history, rag_history
        if llm_final is not None:
            yield rag_time_msg, llm_final, rag_history
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        yield "Error", [["LLM", error_msg]], [["RAG", error_msg]]

with gr.Blocks() as demo:
    gr.Markdown("# LangChain RAG with Ollama Demo\nThis is a simple Gradio app demonstrating Retrieval-Augmented Generation (RAG) using LangChain and Ollama.")
    with gr.Row():
        question = gr.Textbox(label="Enter your question:")
    with gr.Row():
        llm_model = gr.Dropdown(choices=LLM_MODELS, value="llama2", label="LLM Model (for both RAG and LLM-only)")
        btn = gr.Button("Submit", variant="primary")
    with gr.Row():
        query_time = gr.Textbox(label="Query Time")
    gr.Markdown("## LLM-Only Answer (No Wikipedia)")
    llm_chat = gr.Chatbot(label="LLM-Only Answer (No Wikipedia)")
    gr.Markdown("## RAG Answer (Wikipedia-Augmented)")
    rag_chat = gr.Chatbot(label="RAG Answer (Wikipedia-Augmented)")
    def on_submit(q, m):
        # Return the generator directly, not as a tuple/list
        return gradio_interface_stream(q, m)
    question.submit(
        fn=gradio_interface_stream,
        inputs=[question, llm_model],
        outputs=[query_time, llm_chat, rag_chat],
        queue=True
    )
    btn.click(
        fn=gradio_interface_stream,
        inputs=[question, llm_model],
        outputs=[query_time, llm_chat, rag_chat],
        queue=True
    )

demo.queue().launch(share=True)
