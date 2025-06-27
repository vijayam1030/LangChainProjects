#!/bin/bash

# Setup script to pull Ollama models

echo "Setting up Ollama models for LangChain applications..."

# List of models to pull
MODELS=(
    "llama2"
    "qwen3:1.7b" 
    "gemma3:1b"
    "deepseek-r1:1.5b"
    "mistral:7b"
    "phi3:3.8b"
    "tinyllama:1.1b"
    "dolphin3:8b"
    "llama2-uncensored:7b"
)

# Pull each model
for model in "${MODELS[@]}"; do
    echo "Pulling model: $model"
    docker exec langchainprojects-ollama-1 ollama pull "$model"
    if [ $? -eq 0 ]; then
        echo "✅ Successfully pulled $model"
    else
        echo "❌ Failed to pull $model"
    fi
done

echo "Model setup complete!"
echo "You can now use all your LangChain applications."
