#!/usr/bin/env python3
"""
Hybrid streaming app: Fast demo + Real model option
"""
import gradio as gr
import time
import threading
import queue
import random

# Models for demo
DEMO_MODELS = [
    ("demo1", "FastDemo-1"),
    ("demo2", "FastDemo-2"), 
    ("demo3", "FastDemo-3")
]

# Real model (just one for now)
REAL_MODEL = ("tinyllama:1.1b", "TinyLlama")

# Demo responses
DEMO_RESPONSES = [
    "This is a quick response from {}. Based on the question '{}', I can provide detailed analysis.",
    "According to {}, your question about '{}' raises several interesting points to consider.",
    "From {}'s perspective: '{}' is a complex topic that requires careful examination.",
    "{} suggests that '{}' involves multiple factors and considerations.",
    "In {}'s analysis of '{}': This topic has several important aspects worth discussing."
]

def simulate_model_response(question, model_id, model_name, out_queue):
    """Fast simulated model response"""
    try:
        delay = random.uniform(0.5, 2.0)
        response_template = random.choice(DEMO_RESPONSES).format(model_name, question)
        words = response_template.split()
        
        current_response = ""
        for i, word in enumerate(words):
            current_response += word + " "
            out_queue.put((model_name, current_response.strip(), False))
            time.sleep(delay / len(words))
        
        final_response = f"{current_response.strip()}\n\n✅ Complete demo response from {model_name}"
        out_queue.put((model_name, final_response, True))
        
    except Exception as e:
        out_queue.put((model_name, f"Error: {str(e)}", True))

def real_model_response(question, out_queue):
    """Real Ollama model response (non-streaming for reliability)"""
    try:
        from langchain_ollama import OllamaLLM
        
        print(f"🤖 Starting real model: {REAL_MODEL[1]}")
        out_queue.put((REAL_MODEL[1], "🔄 Loading TinyLlama model (this may take 30+ seconds)...", False))
        
        llm = OllamaLLM(
            model=REAL_MODEL[0],
            base_url="http://ollama:11434",
            timeout=60,  # Longer timeout for real model
            temperature=0.3,
            num_predict=100
        )
        
        out_queue.put((REAL_MODEL[1], "🔄 Model loaded, generating response...", False))
        
        # Use invoke instead of streaming for reliability
        result = llm.invoke(question)
        final_response = f"🤖 Real response from {REAL_MODEL[1]}:\n\n{result}\n\n✅ This is a genuine AI response!"
        out_queue.put((REAL_MODEL[1], final_response, True))
        print(f"✅ Real model completed")
        
    except Exception as e:
        error_msg = f"❌ Real model error: {str(e)[:200]}"
        out_queue.put((REAL_MODEL[1], error_msg, True))
        print(f"❌ Real model failed: {e}")

def process_demo_question(question):
    """Fast demo with multiple models"""
    if not question.strip():
        yield ["Please enter a question"] + ["" for _ in DEMO_MODELS]
        return
    
    out_queue = queue.Queue()
    threads = []
    answers = {name: "" for _, name in DEMO_MODELS}
    finished = {name: False for _, name in DEMO_MODELS}
    
    # Start demo models
    for model_id, model_name in DEMO_MODELS:
        thread = threading.Thread(
            target=simulate_model_response,
            args=(question, model_id, model_name, out_queue),
            daemon=True
        )
        thread.start()
        threads.append(thread)
    
    yield [f"🚀 Demo: Processing with {len(DEMO_MODELS)} models..."] + ["🔄 Starting..." for _ in DEMO_MODELS]
    
    start_time = time.time()
    while not all(finished.values()):
        try:
            model_name, answer, is_finished = out_queue.get(timeout=0.1)
            answers[model_name] = answer
            finished[model_name] = is_finished
        except queue.Empty:
            pass
        
        elapsed = time.time() - start_time
        completed = sum(finished.values())
        status = f"⏱️ Demo: {elapsed:.1f}s - {completed}/{len(DEMO_MODELS)} complete - Fast streaming!"
        
        outputs = [status]
        for _, model_name in DEMO_MODELS:
            if answers[model_name]:
                prefix = "✅" if finished[model_name] else "🔄"
                outputs.append(f"{prefix} {answers[model_name]}")
            else:
                outputs.append(f"🔄 {model_name}: Starting...")
        
        yield outputs
        time.sleep(0.1)
        
        if elapsed > 5:
            break
    
    elapsed = time.time() - start_time
    outputs = [f"🎉 Demo complete in {elapsed:.1f}s - This shows perfect streaming!"]
    for _, model_name in DEMO_MODELS:
        outputs.append(f"✅ {answers[model_name]}")
    yield outputs

def process_real_question(question):
    """Real Ollama model (single, reliable)"""
    if not question.strip():
        yield ["Please enter a question", ""]
        return
    
    out_queue = queue.Queue()
    answers = {REAL_MODEL[1]: ""}
    finished = {REAL_MODEL[1]: False}
    
    # Start real model thread
    thread = threading.Thread(
        target=real_model_response,
        args=(question, out_queue),
        daemon=True
    )
    thread.start()
    
    yield [f"🤖 Starting real AI model: {REAL_MODEL[1]}", "🔄 Initializing..."]
    
    start_time = time.time()
    last_update = start_time
    
    while not finished[REAL_MODEL[1]]:
        try:
            model_name, answer, is_finished = out_queue.get(timeout=1.0)
            answers[model_name] = answer
            finished[model_name] = is_finished
            last_update = time.time()
        except queue.Empty:
            pass
        
        elapsed = time.time() - start_time
        
        # Give progress updates
        if elapsed < 30:
            status = f"🤖 Real AI: {elapsed:.0f}s - Please wait (model loading/processing)"
        elif elapsed < 60:
            status = f"🤖 Real AI: {elapsed:.0f}s - Still processing (normal for first use)"
        else:
            status = f"⏰ Real AI: {elapsed:.0f}s - Taking longer than expected"
        
        outputs = [status, answers[REAL_MODEL[1]] if answers[REAL_MODEL[1]] else "🔄 Processing..."]
        yield outputs
        
        # Extended timeout for real models
        if elapsed > 90:
            outputs = [f"⏰ Timeout after {elapsed:.0f}s - Model may need more time to load", 
                      "❌ Try again - subsequent requests should be faster once model is loaded"]
            yield outputs
            break
    
    # Final result
    elapsed = time.time() - start_time
    if finished[REAL_MODEL[1]]:
        outputs = [f"🎉 Real AI completed in {elapsed:.0f}s!", answers[REAL_MODEL[1]]]
    else:
        outputs = [f"⏰ Stopped after {elapsed:.0f}s", answers[REAL_MODEL[1]] or "No response received"]
    
    yield outputs

# Create interface with two modes
with gr.Blocks(title="Demo + Real AI") as app:
    gr.Markdown("# 🚀 Streaming Demo + Real AI")
    gr.Markdown("Choose between **fast demo streaming** or **real AI model** (slower but genuine)")
    
    question_input = gr.Textbox(label="Your Question", placeholder="Ask anything...")
    
    with gr.Tabs():
        with gr.TabItem("⚡ Fast Demo"):
            demo_btn = gr.Button("🚀 Demo Streaming (Fast)", variant="primary", size="lg")
            demo_status = gr.Textbox(label="📊 Demo Status", lines=2)
            demo_outputs = []
            for _, model_name in DEMO_MODELS:
                output = gr.Textbox(label=f"🤖 {model_name}", lines=4)
                demo_outputs.append(output)
        
        with gr.TabItem("🤖 Real AI"):
            real_btn = gr.Button("🤖 Ask Real TinyLlama (Slow)", variant="secondary", size="lg")
            real_status = gr.Textbox(label="📊 Real AI Status", lines=2)
            real_output = gr.Textbox(label=f"🤖 {REAL_MODEL[1]} Response", lines=8)
    
    # Wire up
    demo_btn.click(
        process_demo_question,
        inputs=[question_input],
        outputs=[demo_status] + demo_outputs,
        queue=True
    )
    
    real_btn.click(
        process_real_question,
        inputs=[question_input],
        outputs=[real_status, real_output],
        queue=True
    )

if __name__ == "__main__":
    print("🚀 Starting Hybrid Demo + Real AI App...")
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
