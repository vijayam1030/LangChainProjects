# Docker Multi-Model LangChain Project - Command History
# Session Date: June 27, 2025
# Project: Dockerized streaming multi-model LLM application

## Initial Project Setup and Docker Commands

# Check Docker containers status
docker ps
docker ps | findstr multimodel
docker ps | findstr ollama

# Docker compose operations
docker-compose ps
docker-compose restart multimodel
docker-compose restart
docker-compose build multimodel
docker-compose up -d
docker-compose stop
docker-compose start

# Copy files to Docker containers
docker cp c:\Vijji\LangChainProjects\multimodel\multimodel_fixed.py langchainprojects-multimodel-1:/app/multimodel.py
docker cp c:\Vijji\LangChainProjects\multimodel\streaming_app.py langchainprojects-multimodel-1:/app/streaming_app.py
docker cp c:\Vijji\LangChainProjects\multimodel\debug_streaming.py langchainprojects-multimodel-1:/app/debug_streaming.py
docker cp c:\Vijji\LangChainProjects\multimodel\ollama_debug.py langchainprojects-multimodel-1:/app/ollama_debug.py
docker cp c:\Vijji\LangChainProjects\multimodel\simple_multimodel.py langchainprojects-multimodel-1:/app/simple_multimodel.py
docker cp c:\Vijji\LangChainProjects\multimodel\real_ollama_test.py langchainprojects-multimodel-1:/app/real_ollama_test.py
docker cp c:\Vijji\LangChainProjects\multimodel\hybrid_app.py langchainprojects-multimodel-1:/app/hybrid_app.py

# Docker container inspection
docker exec langchainprojects-multimodel-1 ls -la /app/
docker exec langchainprojects-multimodel-1 ls -la /app/ | grep -E "(streaming_app|multimodel)"
docker exec langchainprojects-multimodel-1 ls -la /app/ | grep real
docker exec langchainprojects-multimodel-1 cat /app/multimodel.py | head -15
docker exec langchainprojects-multimodel-1 head -10 /app/multimodel.py
docker exec langchainprojects-multimodel-1 grep -A 10 "def real_models" /app/multimodel.py
docker exec langchainprojects-multimodel-1 grep -A 20 "def multi_model_stream_interface" /app/multimodel.py

# Docker logs and debugging
docker logs langchainprojects-multimodel-1 --tail 20
docker logs langchainprojects-multimodel-1 --tail 30
docker logs langchainprojects-multimodel-1 --tail 10
docker logs langchainprojects-multimodel-1 --tail 15
docker logs langchainprojects-multimodel-1 --follow --tail 10

# Install packages in Docker containers
docker exec langchainprojects-multimodel-1 pip install langchain_ollama
docker exec langchainprojects-multimodel-1 pip install langchain-ollama

## Ollama Testing and Debugging

# Test Ollama connectivity from host
curl -s http://localhost:11434/api/tags | head -5
curl -X POST http://localhost:11434/api/generate -d "{\"model\":\"tinyllama:1.1b\",\"prompt\":\"Hi\",\"stream\":false}" --max-time 20

# Test Ollama from within multimodel container
docker exec langchainprojects-multimodel-1 curl -s http://ollama:11434/api/tags
docker exec langchainprojects-multimodel-1 curl -s -X POST http://ollama:11434/api/generate -d "{\"model\":\"tinyllama:1.1b\",\"prompt\":\"What is 2+2?\",\"stream\":false}" | head -5

# Check Ollama models
docker exec langchainprojects-ollama-1 ollama list

# Test Python LangChain connection
docker exec langchainprojects-multimodel-1 python -c "from langchain_ollama import OllamaLLM; llm = OllamaLLM(model='tinyllama:1.1b', base_url='http://ollama:11434', timeout=10); print('Testing:', llm.invoke('What is 1+1?'))"

## WSL and Node.js Setup (for Claude Code installation)

# Switch to WSL
wsl

# Check environment in WSL
echo $OSTYPE

# Check current npm installation
which npm
npm --version

# Install NVM (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash

# Load NVM and install Node.js
export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm install --lts
nvm use --lts

# Verify WSL-native npm
which npm && npm --version

# Install Claude Code
npm install -g @anthropic-ai/claude-code

# Test Claude Code installation
claude-code --version
claude --version
claude --help

# Find Claude Code installation
find ~/.nvm -name "claude-code" 2>/dev/null
ls ~/.nvm/versions/node/v22.17.0/bin/ | grep claude

# Reload shell configuration
source ~/.bashrc

## Alternative Node.js Installation Attempts (WSL)

# Try APT package manager (had conflicts)
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo apt update && sudo apt install -y nodejs npm

## Environment and System Information

# Check OS type
echo $OSTYPE

# Directory listings and file operations
ls -la
cd /mnt/c/Vijji/LangChainProjects

## Docker Compose File Updates

# The docker-compose.yml command field was updated multiple times:
# - Originally: command: ["python", "multimodel/multimodel.py"]
# - Changed to: command: ["python", "streaming_app.py"]
# - Changed to: command: ["python", "debug_streaming.py"]
# - Changed to: command: ["python", "ollama_debug.py"]
# - Changed to: command: ["python", "simple_multimodel.py"]
# - Changed to: command: ["python", "real_ollama_test.py"]
# - Finally: command: ["python", "hybrid_app.py"]

## Application URLs for Testing

# Open applications in browser
# http://localhost:7860  (Main multimodel app)
# http://localhost:8501  (RAG Streamlit app)
# http://localhost:7861  (RAG Gradio app)

## Key Files Created During Session

# 1. streaming_app.py - Clean streaming app with LangChain integration
# 2. debug_streaming.py - Basic streaming test without LLM calls
# 3. ollama_debug.py - Ollama connection diagnostic tool
# 4. simple_multimodel.py - Fast demo with simulated responses
# 5. real_ollama_test.py - Optimized real Ollama model test
# 6. hybrid_app.py - Combined fast demo + real AI tabs
# 7. multimodel_fixed.py - LangChain-based multimodel implementation

## Problem Resolution Workflow

# 1. Initial issue: Streaming not working, 60-second timeouts
# 2. Diagnostic approach: Isolated variables (Gradio, threading, Ollama)
# 3. Root cause: Ollama models too slow for real-time streaming
# 4. Solution: Created hybrid app with fast demo + real AI options
# 5. Final result: Working streaming demo + functional real model access

## Key Learnings

# - Ollama models can take 30+ seconds for first inference (cold start)
# - LangChain OllamaLLM is better than raw HTTP requests
# - Streaming works perfectly when response time is reasonable
# - WSL requires native Node.js installation for Claude Code
# - Docker file copying needs exact container names
# - Multiple restart approaches: docker-compose restart vs docker-compose build

## Environment Variables Used

# OLLAMA_BASE_URL=http://ollama:11434
# GRADIO_SERVER_NAME=0.0.0.0
# GRADIO_SERVER_PORT=7860
# MAX_WORKERS=4

## Additional Commands Used

# Check Python environment and packages
docker exec langchainprojects-multimodel-1 python --version
docker exec langchainprojects-multimodel-1 pip list | grep -E "(gradio|langchain|ollama)"

# File editing and testing commands
docker exec langchainprojects-multimodel-1 cat /app/requirements.txt
docker exec langchainprojects-multimodel-1 python -c "import gradio; print(gradio.__version__)"

# Model setup commands (Ollama)
docker exec langchainprojects-ollama-1 ollama pull tinyllama:1.1b
docker exec langchainprojects-ollama-1 ollama pull qwen2.5:0.5b
docker exec langchainprojects-ollama-1 ollama pull qwen2.5:1.5b

# Test individual Python files
docker exec langchainprojects-multimodel-1 python /app/debug_streaming.py
docker exec langchainprojects-multimodel-1 python /app/streaming_test.py

# Container resource inspection
docker stats langchainprojects-multimodel-1 --no-stream
docker stats langchainprojects-ollama-1 --no-stream

# Network connectivity tests
docker exec langchainprojects-multimodel-1 ping ollama
docker exec langchainprojects-multimodel-1 nslookup ollama

## Final Architecture

# 1. Fast Demo Tab: 3 simulated models with perfect streaming (1-3 seconds)
# 2. Real AI Tab: Single TinyLlama model with extended timeout (30-90 seconds)
# 3. Docker containers: multimodel app + Ollama service + RAG apps
# 4. LangChain integration for reliable model communication
# 5. Gradio interface with tabbed layout for different use cases

## Session Summary

# Total Duration: ~4-5 hours
# Primary Goal: Fix streaming and parallel execution in Docker
# Result: Successfully created working streaming demo + real AI hybrid app
# Key Innovation: Separated fast demo from slow real models for better UX
# Documentation: Complete command history for reproducibility
