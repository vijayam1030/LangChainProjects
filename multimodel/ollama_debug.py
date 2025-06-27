#!/usr/bin/env python3
"""
Ollama connection diagnostic test
"""
import gradio as gr
import time
import requests
import json

def test_ollama_http():
    """Test direct HTTP connection to Ollama"""
    try:
        # Test 1: Check if Ollama is reachable
        url = "http://ollama:11434/api/tags"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            yield f"✅ Ollama HTTP connection works!\nAvailable models: {model_names}"
        else:
            yield f"❌ Ollama HTTP error: {response.status_code}"
            return
        
        # Test 2: Try a simple generate call
        yield "🔄 Testing simple generation..."
        gen_url = "http://ollama:11434/api/generate"
        payload = {
            "model": "tinyllama:1.1b",
            "prompt": "Say hello",
            "stream": False
        }
        
        gen_response = requests.post(gen_url, json=payload, timeout=10)
        if gen_response.status_code == 200:
            result = gen_response.json()
            answer = result.get('response', 'No response')
            yield f"✅ Direct HTTP generation works!\nTinyLlama says: {answer}"
        else:
            yield f"❌ Generation failed: {gen_response.status_code}"
            
    except Exception as e:
        yield f"❌ HTTP test failed: {str(e)}"

def test_langchain_ollama():
    """Test LangChain Ollama connection"""
    try:
        yield "🔄 Testing LangChain OllamaLLM..."
        
        from langchain_ollama import OllamaLLM
        
        # Test with very short timeout
        llm = OllamaLLM(
            model="tinyllama:1.1b",
            base_url="http://ollama:11434",
            timeout=10
        )
        
        yield "🔄 Calling LangChain invoke..."
        result = llm.invoke("Say hello briefly")
        yield f"✅ LangChain works!\nResult: {result}"
        
    except Exception as e:
        yield f"❌ LangChain test failed: {str(e)}"

def test_streaming():
    """Test LangChain streaming"""
    try:
        yield "🔄 Testing LangChain streaming..."
        
        from langchain_ollama import OllamaLLM
        
        llm = OllamaLLM(
            model="tinyllama:1.1b",
            base_url="http://ollama:11434",
            timeout=15
        )
        
        answer = ""
        chunk_count = 0
        for chunk in llm.stream("Count to 3"):
            answer += chunk
            chunk_count += 1
            yield f"🔄 Streaming chunk {chunk_count}: {answer}"
            if chunk_count > 10:  # Safety limit
                break
        
        yield f"✅ Streaming works! Final: {answer}"
        
    except Exception as e:
        yield f"❌ Streaming test failed: {str(e)}"

# Debug interface
with gr.Blocks(title="Ollama Debug") as app:
    gr.Markdown("# 🔧 Ollama Connection Debug")
    
    with gr.Row():
        http_btn = gr.Button("Test HTTP")
        langchain_btn = gr.Button("Test LangChain")
        stream_btn = gr.Button("Test Streaming")
    
    output = gr.Textbox(label="Test Results", lines=10)
    
    http_btn.click(test_ollama_http, outputs=[output])
    langchain_btn.click(test_langchain_ollama, outputs=[output])
    stream_btn.click(test_streaming, outputs=[output])

if __name__ == "__main__":
    print("🔧 Starting Ollama debug app...")
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
