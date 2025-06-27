#!/usr/bin/env python3
"""
Quick test to check if models respond within the container
"""
import os
import time
from langchain_ollama import OllamaLLM

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
print(f"Testing Ollama at: {OLLAMA_BASE_URL}")

# Test just one fast model
try:
    print("🧪 Testing tinyllama:1.1b (should be fast)...")
    start_time = time.time()
    
    llm = OllamaLLM(
        model="tinyllama:1.1b", 
        base_url=OLLAMA_BASE_URL,
        timeout=30
    )
    
    result = llm.invoke("What is 2+2?")
    elapsed = time.time() - start_time
    
    print(f"✅ SUCCESS: Got response in {elapsed:.1f}s")
    print(f"📝 Response: {result}")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
