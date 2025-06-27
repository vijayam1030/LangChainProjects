# Configuration settings for LangChain projects

import os
from typing import Dict, List, Tuple

class AppConfig:
    """Application configuration settings"""
    
    # Ollama settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Default model settings
    DEFAULT_MODEL = "llama2"
    DEFAULT_EMBEDDING_MODEL = "llama2"
    
    # UI settings
    GRADIO_SHARE = os.getenv("GRADIO_SHARE", "true").lower() == "true"
    GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    
    # RAG settings
    WIKIPEDIA_MAX_DOCS = 5
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    
    # Performance settings
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
    STREAMING_TIMEOUT = 0.1

class ModelConfig:
    """Model configurations and metadata"""
    
    # Available models with metadata
    MODELS: Dict[str, Dict] = {
        "llama2": {
            "name": "Llama 2",
            "size": "7B",
            "type": "general",
            "description": "General purpose language model"
        },
        "qwen3:1.7b": {
            "name": "Qwen 3",
            "size": "1.7B",
            "type": "small",
            "description": "Fast, lightweight model"
        },
        "gemma3:1b": {
            "name": "Gemma 3",
            "size": "1B",
            "type": "small",
            "description": "Efficient small model"
        },
        "deepseek-r1:1.5b": {
            "name": "DeepSeek R1",
            "size": "1.5B",
            "type": "reasoning",
            "description": "Reasoning-focused model"
        },
        "mistral:7b": {
            "name": "Mistral",
            "size": "7B",
            "type": "general",
            "description": "High-quality general model"
        },
        "phi3:3.8b": {
            "name": "Phi 3",
            "size": "3.8B",
            "type": "medium",
            "description": "Balanced performance model"
        },
        "tinyllama:1.1b": {
            "name": "TinyLlama",
            "size": "1.1B",
            "type": "tiny",
            "description": "Ultra-fast tiny model"
        },
        "dolphin3:8b": {
            "name": "Dolphin 3",
            "size": "8B",
            "type": "uncensored",
            "description": "Uncensored general model"
        },
        "llama2-uncensored:7b": {
            "name": "Llama 2 Uncensored",
            "size": "7B",
            "type": "uncensored",
            "description": "Uncensored Llama 2"
        }
    }
    
    @classmethod
    def get_model_list(cls) -> List[Tuple[str, str]]:
        """Get list of models for UI dropdowns"""
        return [(model_id, f"{config['name']} ({config['size']})") 
                for model_id, config in cls.MODELS.items()]
    
    @classmethod
    def get_models_by_type(cls, model_type: str) -> List[str]:
        """Get models filtered by type"""
        return [model_id for model_id, config in cls.MODELS.items() 
                if config.get('type') == model_type]
    
    @classmethod
    def get_small_models(cls) -> List[str]:
        """Get small/fast models for quick testing"""
        return cls.get_models_by_type('small') + cls.get_models_by_type('tiny')

# Environment-specific settings
class DevConfig(AppConfig):
    """Development environment settings"""
    DEBUG = True
    GRADIO_SHARE = False

class ProdConfig(AppConfig):
    """Production environment settings"""
    DEBUG = False
    GRADIO_SHARE = True
