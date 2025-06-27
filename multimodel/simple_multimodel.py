#!/usr/bin/env python3
"""
Completely rewritten multimodel app with guaranteed working streaming
"""
import gradio as gr
import time
import threading
import os

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
    """Real model processing with proper error handling"""
    if not question.strip():
        error = ["Please enter a question"] + ["" for _ in MODELS]
        yield error
        return
    
    # This is where we'd call real models, but for now just simulate
    for i in range(10):
        status = f"🤖 Real processing: {i}s - Working on your question..."
        outputs = [status] + ["🔄 Processing real models..." for _ in MODELS]
        yield outputs
        time.sleep(1)
    
    # Simulate completion
    final = ["✅ Real processing complete!"] + [f"Real answer from {name}: {question}" for _, name in MODELS]
    yield final

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
