# Shared vector store utilities

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import CharacterTextSplitter
from typing import List, Tuple, Optional
from functools import lru_cache

class VectorStoreManager:
    """Centralized vector store management"""
    
    @staticmethod
    @lru_cache(maxsize=10)
    def get_wikipedia_docs_and_vectorstore(
        question: str, 
        lang: str = "en", 
        max_docs: int = 5,
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ) -> Tuple[List, Optional[FAISS]]:
        """
        Get Wikipedia documents and create vector store with caching
        
        Args:
            question: Search query for Wikipedia
            lang: Language for Wikipedia search
            max_docs: Maximum number of documents to load
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            Tuple of (wiki_docs, vectorstore)
        """
        try:
            # Load Wikipedia documents
            loader = WikipediaLoader(query=question, lang=lang, load_max_docs=max_docs)
            wiki_docs = loader.load()
            
            if not wiki_docs:
                return [], None
            
            # Split documents into chunks
            docs = [doc.page_content for doc in wiki_docs]
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            split_docs = text_splitter.create_documents(docs)
            
            # Create embeddings and vector store
            embeddings = OllamaEmbeddings(model="llama2")
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            
            return wiki_docs, vectorstore
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return [], None
    
    @staticmethod
    def get_relevant_context(vectorstore: FAISS, question: str, k: int = 4) -> str:
        """Get relevant context from vector store"""
        if not vectorstore:
            return ""
        
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            relevant_docs = retriever.get_relevant_documents(question)
            return "\n\n".join([doc.page_content for doc in relevant_docs])
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return ""

class RAGPrompts:
    """Centralized prompt templates for RAG"""
    
    DEFAULT_RAG_TEMPLATE = """Answer the question using ONLY the following context from Wikipedia.
If the answer is not in the context, say 'I don't know.'

Context:
{context}

Question: {question}

Answer:"""
    
    @classmethod
    def format_rag_prompt(cls, context: str, question: str, template: str = None) -> str:
        """Format RAG prompt with context and question"""
        if template is None:
            template = cls.DEFAULT_RAG_TEMPLATE
        
        return template.format(context=context, question=question)
