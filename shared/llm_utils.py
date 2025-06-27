# Shared LLM utilities for all LangChain projects

from langchain_ollama import OllamaLLM
from typing import List, Dict, Any
import time

class LLMManager:
    """Centralized LLM management for all applications"""
    
    # Common model configurations
    MODELS = {
        "llama2": {"name": "Llama 2", "size": "7B"},
        "qwen3:1.7b": {"name": "Qwen 3", "size": "1.7B"},
        "gemma3:1b": {"name": "Gemma 3", "size": "1B"},
        "deepseek-r1:1.5b": {"name": "DeepSeek R1", "size": "1.5B"},
        "mistral:7b": {"name": "Mistral", "size": "7B"},
        "phi3:3.8b": {"name": "Phi 3", "size": "3.8B"},
        "tinyllama:1.1b": {"name": "TinyLlama", "size": "1.1B"},
        "dolphin3:8b": {"name": "Dolphin 3", "size": "8B"},
        "llama2-uncensored:7b": {"name": "Llama 2 Uncensored", "size": "7B"},
    }
    
    @classmethod
    def get_available_models(cls) -> List[tuple]:
        """Get list of available models in (id, display_name) format"""
        return [(model_id, f"{config['name']} {config['size']}") 
                for model_id, config in cls.MODELS.items()]
    
    @classmethod
    def create_llm(cls, model_id: str, streaming: bool = False, **kwargs) -> OllamaLLM:
        """Create an OllamaLLM instance with consistent configuration"""
        return OllamaLLM(
            model=model_id,
            streaming=streaming,
            **kwargs
        )
    
    @classmethod
    def get_response(cls, model_id: str, prompt: str, streaming: bool = False) -> str:
        """Get response from a model with error handling"""
        try:
            llm = cls.create_llm(model_id, streaming=streaming)
            if streaming:
                return llm.stream(prompt)
            else:
                return llm.invoke(prompt)
        except Exception as e:
            return f"Error with {model_id}: {str(e)}"

def measure_time(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result, elapsed_time
    return wrapper

def format_query_time(elapsed_seconds: float) -> str:
    """Format elapsed time in a user-friendly way"""
    return f"⏱️ Query time: {elapsed_seconds:.2f} seconds"
