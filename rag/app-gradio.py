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
LLM_MODELS = ["llama2",  "qwen3:1.7b", "gemma3:1b", "deepseek-r1:1.5b", "mistral:7b"]

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
        [["RAG", rag_ans]],
        [["LLM", llm_ans]]
    )

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
        return gradio_interface(q, m)
    question.submit(
        on_submit,
        inputs=[question, llm_model],
        outputs=[query_time, rag_chat, llm_chat]
    )
    btn.click(
        on_submit,
        inputs=[question, llm_model],
        outputs=[query_time, rag_chat, llm_chat]
    )

demo.queue().launch(share=True)
