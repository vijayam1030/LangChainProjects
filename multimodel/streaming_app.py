#!/usr/bin/env python3
"""
Clean streaming multimodel app for Docker deployment
Fast timeouts, clear streaming, parallel execution
"""
import gradio as gr
from langchain_ollama import OllamaLLM
import time
import threading
import queue
import os

# Ollama connection
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# Fast models only
MODELS = [
    ("tinyllama:1.1b", "TinyLlama"),
    ("qwen3:1.7b", "Qwen3"),
    ("gemma3:1b", "Gemma3")
]

def stream_model_response(question, model_id, model_label, out_queue):
    """Stream response from a single model"""
    try:
        print(f"🚀 Starting {model_label}...")
        llm = OllamaLLM(
            model=model_id,
            base_url=OLLAMA_BASE_URL,
            timeout=15  # Short timeout - 15 seconds
        )
        
        # Stream the response
        answer = ""
        for chunk in llm.stream(question):
            answer += chunk
            out_queue.put((model_label, answer, False))  # False = not finished
        
        # Mark as complete
        out_queue.put((model_label, answer, True))  # True = finished
        print(f"✅ {model_label} completed")
        
    except Exception as e:
        error_msg = f"Error: {str(e)[:100]}"
        out_queue.put((model_label, error_msg, True))
        print(f"❌ {model_label} failed: {e}")

def process_question(question):
    """Process question with all models in parallel with streaming"""
    if not question.strip():
        yield ["Please enter a question"] + ["" for _ in MODELS]
        return
    
    # Setup
    out_queue = queue.Queue()
    threads = []
    answers = {label: "" for _, label in MODELS}
    finished = {label: False for _, label in MODELS}
    
    # Start all models in parallel
    for model_id, model_label in MODELS:
        thread = threading.Thread(
            target=stream_model_response,
            args=(question, model_id, model_label, out_queue),
            daemon=True
        )
        thread.start()
        threads.append(thread)
        print(f"🔄 Started thread for {model_label}")
    
    # Initial output
    yield [f"🚀 Started {len(MODELS)} models in parallel..."] + ["🔄 Starting..." for _ in MODELS]
    
    # Stream updates
    start_time = time.time()
    update_count = 0
    
    while not all(finished.values()):
        try:
            # Get update from queue (non-blocking with short timeout)
            model_label, answer, is_finished = out_queue.get(timeout=0.2)
            answers[model_label] = answer
            finished[model_label] = is_finished
            update_count += 1
            
        except queue.Empty:
            pass
        
        # Build current status
        elapsed = time.time() - start_time
        completed_count = sum(finished.values())
        status = f"⏱️ {elapsed:.1f}s - Updates: {update_count} - Complete: {completed_count}/{len(MODELS)}"
        
        # Build outputs
        outputs = [status]
        for _, model_label in MODELS:
            if finished[model_label]:
                outputs.append(f"✅ {model_label}:\n{answers[model_label]}")
            elif answers[model_label]:
                outputs.append(f"🔄 {model_label}:\n{answers[model_label]}...")
            else:
                outputs.append(f"🔄 {model_label}: Processing...")
        
        yield outputs
        
        # Safety timeout - 30 seconds total
        if elapsed > 30:
            print("⏰ Global timeout reached")
            break
        
        # Small delay to prevent UI overload
        time.sleep(0.1)
    
    # Final results
    elapsed = time.time() - start_time
    final_completed = sum(finished.values())
    
    outputs = [f"🎉 Complete! {final_completed}/{len(MODELS)} models finished in {elapsed:.1f}s"]
    for _, model_label in MODELS:
        if finished[model_label]:
            outputs.append(f"✅ {model_label}:\n{answers[model_label]}")
        else:
            outputs.append(f"⏰ {model_label}: Timed out")
    
    yield outputs
    print(f"🏁 Processing complete: {final_completed}/{len(MODELS)} models")

# Create Gradio interface
with gr.Blocks(title="Streaming Multi-Model App") as app:
    gr.Markdown("# 🚀 Streaming Multi-Model LLM App")
    gr.Markdown("Enter a question and watch real-time streaming responses from multiple models!")
    
    # Input
    with gr.Row():
        question_input = gr.Textbox(
            label="Your Question", 
            placeholder="Ask anything...",
            scale=4
        )
        submit_btn = gr.Button("🚀 Submit", variant="primary", scale=1)
    
    # Status
    status_output = gr.Textbox(label="📊 Status", lines=2)
    
    # Model outputs
    model_outputs = []
    for _, model_label in MODELS:
        output = gr.Textbox(
            label=f"🤖 {model_label}", 
            lines=4,
            max_lines=8
        )
        model_outputs.append(output)
    
    # Wire up the interface
    submit_btn.click(
        process_question,
        inputs=[question_input],
        outputs=[status_output] + model_outputs,
        queue=True
    )
    
    question_input.submit(
        process_question,
        inputs=[question_input],
        outputs=[status_output] + model_outputs,
        queue=True
    )

if __name__ == "__main__":
    print("🚀 Starting Streaming Multi-Model App...")
    print(f"📡 Ollama URL: {OLLAMA_BASE_URL}")
    print(f"🤖 Models: {[label for _, label in MODELS]}")
    
    app.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        show_tips=False
    )
