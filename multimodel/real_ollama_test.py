#!/usr/bin/env python3
"""
Optimized real Ollama model app with short timeouts
"""
import gradio as gr
from langchain_ollama import OllamaLLM
import time
import threading
import queue

# Start with just the fastest model
MODELS = [
    ("tinyllama:1.1b", "TinyLlama")
]

def stream_real_model(question, model_id, model_name, out_queue):
    """Stream from a real Ollama model with optimizations"""
    try:
        print(f"🚀 Starting {model_name}...")
        
        # Create LLM with aggressive timeout settings
        llm = OllamaLLM(
            model=model_id,
            base_url="http://ollama:11434",
            timeout=10,  # Very short timeout
            temperature=0.1,  # Faster inference
            top_p=0.9,
            num_predict=50  # Limit response length for speed
        )
        
        # Try streaming first
        try:
            answer = ""
            chunk_count = 0
            for chunk in llm.stream(question):
                answer += chunk
                chunk_count += 1
                out_queue.put((model_name, answer, False))
                
                # Safety limit to prevent infinite streaming
                if chunk_count > 20:
                    break
            
            # Mark as complete
            out_queue.put((model_name, answer, True))
            print(f"✅ {model_name} streaming completed")
            
        except Exception as stream_error:
            print(f"⚠️ Streaming failed for {model_name}, trying invoke: {stream_error}")
            
            # Fallback to regular invoke
            answer = llm.invoke(question)
            out_queue.put((model_name, answer, True))
            print(f"✅ {model_name} invoke completed")
            
    except Exception as e:
        error_msg = f"❌ Error: {str(e)[:100]}"
        out_queue.put((model_name, error_msg, True))
        print(f"❌ {model_name} failed: {e}")

def process_real_question(question):
    """Process with real Ollama models"""
    if not question.strip():
        yield ["Please enter a question"] + ["" for _ in MODELS]
        return
    
    # Setup
    out_queue = queue.Queue()
    threads = []
    answers = {name: "" for _, name in MODELS}
    finished = {name: False for _, name in MODELS}
    
    # Start model threads
    for model_id, model_name in MODELS:
        thread = threading.Thread(
            target=stream_real_model,
            args=(question, model_id, model_name, out_queue),
            daemon=True
        )
        thread.start()
        threads.append(thread)
        print(f"🔄 Started real model thread: {model_name}")
    
    # Initial output
    yield [f"🤖 Starting real model: {MODELS[0][1]}..."] + ["🔄 Loading model..." for _ in MODELS]
    
    # Stream updates with shorter intervals
    start_time = time.time()
    update_count = 0
    last_update_time = start_time
    
    while not all(finished.values()):
        try:
            # Get updates from queue
            model_name, answer, is_finished = out_queue.get(timeout=0.2)
            answers[model_name] = answer
            finished[model_name] = is_finished
            update_count += 1
            last_update_time = time.time()
            
        except queue.Empty:
            # Check if we should give a progress update
            current_time = time.time()
            if current_time - last_update_time > 2.0:  # Update every 2 seconds even without new data
                last_update_time = current_time
        
        # Build status
        elapsed = time.time() - start_time
        completed_count = sum(finished.values())
        
        if update_count > 0:
            status = f"⏱️ {elapsed:.1f}s - 🔄 Streaming... Updates: {update_count}"
        else:
            status = f"⏱️ {elapsed:.1f}s - 🔄 Model loading... (this may take 10-30s first time)"
        
        # Build outputs
        outputs = [status]
        for _, model_name in MODELS:
            if answers[model_name]:
                prefix = "✅" if finished[model_name] else "🔄"
                outputs.append(f"{prefix} {model_name}:\n{answers[model_name]}")
            else:
                outputs.append(f"🔄 {model_name}: Loading model into memory...")
        
        yield outputs
        
        # Shorter overall timeout - 20 seconds
        if elapsed > 20:
            print("⏰ Timeout reached")
            break
        
        time.sleep(0.2)  # Faster updates
    
    # Final results
    elapsed = time.time() - start_time
    completed_count = sum(finished.values())
    
    if completed_count > 0:
        status = f"🎉 Complete! {completed_count}/{len(MODELS)} models responded in {elapsed:.1f}s"
    else:
        status = f"⏰ Timeout after {elapsed:.1f}s - Model may be loading for first time"
    
    outputs = [status]
    for _, model_name in MODELS:
        if answers[model_name]:
            outputs.append(f"✅ {model_name}:\n{answers[model_name]}")
        else:
            outputs.append(f"⏰ {model_name}: No response (try again - model may be loading)")
    
    yield outputs

# Minimal interface focused on real models
with gr.Blocks(title="Real Ollama Model Test") as app:
    gr.Markdown("# 🤖 Real Ollama Model Test")
    gr.Markdown("Testing with TinyLlama (fastest model). First run may be slow as model loads into memory.")
    
    with gr.Row():
        question_input = gr.Textbox(
            label="Question", 
            placeholder="Ask a simple question (e.g., 'What is 2+2?')",
            scale=4
        )
        submit_btn = gr.Button("🤖 Ask TinyLlama", variant="primary", scale=1)
    
    status_output = gr.Textbox(label="📊 Status", lines=2)
    
    model_outputs = []
    for _, model_name in MODELS:
        output = gr.Textbox(label=f"🤖 {model_name}", lines=6)
        model_outputs.append(output)
    
    # Wire up
    submit_btn.click(
        process_real_question,
        inputs=[question_input],
        outputs=[status_output] + model_outputs,
        queue=True
    )
    
    question_input.submit(
        process_real_question,
        inputs=[question_input],
        outputs=[status_output] + model_outputs,
        queue=True
    )

if __name__ == "__main__":
    print("🤖 Starting Real Ollama Model Test...")
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
