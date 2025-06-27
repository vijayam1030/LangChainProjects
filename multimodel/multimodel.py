#!/usr/bin/env python3
"""
Multimodel app with working parallel LLM calls and streaming
"""
import gradio as gr
import time
import threading
import os

# Add LangChain import for real model calls
from langchain_ollama import OllamaLLM

# Ollama connection
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# Only the fastest models
MODELS = [
    ("tinyllama:1.1b", "TinyLlama"),
    ("qwen3:1.7b", "Qwen3"),
    ("gemma3:1b", "Gemma3")
]

def simple_test():
    """Test streaming works at all"""
    for i in range(5):
        outputs = [f"⚡ Test {i+1}/5: Streaming works!"] + [f"Test output {i+1}" for _ in MODELS]
        yield outputs
        time.sleep(0.5)
    
    final = ["✅ Streaming test completed!"] + ["Final test result" for _ in MODELS]
    yield final

def threaded_demo():
    """Demo with real threading but fake results"""
    results = [""] * len(MODELS)
    completed = [False] * len(MODELS)
    
    def fake_model(index, delay):
        time.sleep(delay)
        results[index] = f"Demo result from {MODELS[index][1]} after {delay}s"
        completed[index] = True
    
    # Start threads
    for i, (_, name) in enumerate(MODELS):
        delay = (i + 1) * 1.5  # 1.5, 3, 4.5 seconds
        thread = threading.Thread(target=fake_model, args=(i, delay), daemon=True)
        thread.start()
    
    # Stream updates
    for update in range(20):  # 10 seconds of updates
        status = f"🔄 Demo: {update * 0.5:.1f}s - {sum(completed)}/{len(MODELS)} complete"
        
        outputs = [status]
        for i, (_, name) in enumerate(MODELS):
            if completed[i]:
                outputs.append(f"✅ {results[i]}")
            else:
                outputs.append("🔄 Processing...")
        
        yield outputs
        time.sleep(0.5)
        
        if all(completed):
            break
    
    # Final
    final = [f"🎉 Demo complete! All {len(MODELS)} threads finished"] + [f"✅ {results[i]}" for i in range(len(MODELS))]
    yield final

def real_models(question):
    """Real parallel LLM processing with actual Ollama models"""
    if not question.strip():
        error = ["Please enter a question"] + ["" for _ in MODELS]
        yield error
        return
    
    # Shared state for parallel results
    results = [""] * len(MODELS)
    completed = [False] * len(MODELS)
    errors = [None] * len(MODELS)
    result_lock = threading.Lock()
    
    def call_model(index, model_id, model_name):
        """Call a real LLM model in a thread"""
        try:
            print(f"🚀 Starting {model_name}...")
            llm = OllamaLLM(
                model=model_id,
                base_url=OLLAMA_BASE_URL,
                timeout=30  # 30 second timeout
            )
            result = llm.invoke(question)
            
            with result_lock:
                results[index] = result
                completed[index] = True
                print(f"✅ {model_name} completed")
                
        except Exception as e:
            with result_lock:
                errors[index] = str(e)[:100] + "..."
                completed[index] = True
                print(f"❌ {model_name} failed: {e}")
    
    # Start all models in parallel threads
    threads = []
    for i, (model_id, model_name) in enumerate(MODELS):
        thread = threading.Thread(
            target=call_model, 
            args=(i, model_id, model_name), 
            daemon=True
        )
        thread.start()
        threads.append(thread)
    
    # Initial status
    outputs = ["🚀 Starting all 3 models in parallel..."] + ["⏳ Starting..." for _ in MODELS]
    yield outputs
    time.sleep(0.1)
    
    # Show all running
    outputs = ["🔄 All models running in parallel..."] + ["🔄 Processing..." for _ in MODELS]
    yield outputs
    
    # Stream updates as results come in
    start_time = time.time()
    last_completed_count = 0
    
    while not all(completed):
        time.sleep(0.5)  # Check every 0.5 seconds
        elapsed = time.time() - start_time
        
        with result_lock:
            current_completed = sum(completed)
            current_results = results.copy()
            current_errors = errors.copy()
        
        # Build status message
        if current_completed > last_completed_count:
            status = f"⏱️ {elapsed:.1f}s - NEW RESULT! {current_completed}/{len(MODELS)} complete"
            last_completed_count = current_completed
        else:
            status = f"🔄 {elapsed:.1f}s - {current_completed}/{len(MODELS)} complete - Working..."
        
        # Build outputs
        outputs = [status]
        for i, (_, model_name) in enumerate(MODELS):
            if completed[i]:
                if current_errors[i]:
                    outputs.append(f"❌ Error from {model_name}: {current_errors[i]}")
                else:
                    outputs.append(f"✅ {current_results[i]}")
            else:
                outputs.append("🔄 Processing...")
        
        yield outputs
        
        # Safety timeout after 60 seconds
        if elapsed > 60:
            break
    
    # Final results
    elapsed = time.time() - start_time
    with result_lock:
        final_completed = sum(completed)
        
    outputs = [f"🎉 Parallel processing complete! {final_completed}/{len(MODELS)} models finished in {elapsed:.1f}s"]
    for i, (_, model_name) in enumerate(MODELS):
        if completed[i]:
            if errors[i]:
                outputs.append(f"❌ Error from {model_name}: {errors[i]}")
            else:
                outputs.append(f"✅ {results[i]}")
        else:
            outputs.append(f"⏰ {model_name}: Timed out after 60s")
    
    yield outputs

# Create interface
with gr.Blocks(title="Multimodel Streaming Test") as app:
    gr.Markdown("# 🧪 Multimodel Streaming Test\nTesting parallel processing and streaming")
    
    # Input
    question = gr.Textbox(label="Question", placeholder="Enter your question...")
    
    # Buttons
    with gr.Row():
        test_btn = gr.Button("⚡ Test Streaming", variant="secondary")
        demo_btn = gr.Button("🎭 Demo Threading", variant="secondary")  
        real_btn = gr.Button("🤖 Real Models", variant="primary")
    
    # Outputs
    status = gr.Textbox(label="Status", lines=2)
    outputs = []
    for _, name in MODELS:
        outputs.append(gr.Textbox(label=f"{name} Result", lines=3))
    
    # Wire up buttons
    test_btn.click(simple_test, inputs=[], outputs=[status] + outputs)
    demo_btn.click(threaded_demo, inputs=[], outputs=[status] + outputs)
    real_btn.click(real_models, inputs=[question], outputs=[status] + outputs)

if __name__ == "__main__":
    print("🚀 Starting multimodel app...")
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
