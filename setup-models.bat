@echo off
REM Windows batch script to pull Ollama models

echo Setting up Ollama models for LangChain applications...

REM List of models to pull
set MODELS=llama2 qwen3:1.7b gemma3:1b deepseek-r1:1.5b mistral:7b phi3:3.8b tinyllama:1.1b dolphin3:8b llama2-uncensored:7b

for %%m in (%MODELS%) do (
    echo Pulling model: %%m
    docker exec langchainprojects-ollama-1 ollama pull %%m
    if errorlevel 1 (
        echo ❌ Failed to pull %%m
    ) else (
        echo ✅ Successfully pulled %%m
    )
)

echo Model setup complete!
echo You can now use all your LangChain applications.
