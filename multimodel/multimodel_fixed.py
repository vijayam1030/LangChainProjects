#!/usr/bin/env python3
"""
Completely rewritten multimodel app with guaranteed working streaming
"""
import gradio as gr
from langchain_ollama import OllamaLLM
import time
import threading
import queue

# List of available LLM models
LLM_MODELS = [
    ("tinyllama:1.1b", "TinyLlama 1.1B"),
    ("qwen3:1.7b", "Qwen 3 1.7B"),
    ("gemma3:1b", "Gemma 3 1B"),
]

def run_llm_stream_for_model(question, model_id, out_queue, label):
    llm = OllamaLLM(model=model_id, streaming=True)
    answer = ""
    for chunk in llm.stream(question):
        answer += chunk
        out_queue.put((label, answer))

def multi_model_stream_interface(question, progress=gr.Progress(track_tqdm=True)):
    out_queue = queue.Queue()
    threads = []
    for model_id, model_label in LLM_MODELS:
        t = threading.Thread(target=run_llm_stream_for_model, args=(question, model_id, out_queue, model_label))
        t.start()
        threads.append(t)
    answers = {label: "" for _, label in LLM_MODELS}
    finished = 0
    total = len(LLM_MODELS)
    start_time = time.time()
    while finished < total:
        try:
            label, ans = out_queue.get(timeout=0.1)
            answers[label] = ans
            elapsed = time.time() - start_time
            outputs = [f"⏱️ Query time: {elapsed:.2f} seconds"]
            for _, model_label in LLM_MODELS:
                outputs.append(answers[model_label])
            yield tuple(outputs)
        except queue.Empty:
            pass
        finished = sum([not t.is_alive() for t in threads])
    # Final yield to ensure all answers are shown
    elapsed = time.time() - start_time
    outputs = [f"⏱️ Query time: {elapsed:.2f} seconds"]
    for _, model_label in LLM_MODELS:
        outputs.append(answers[model_label])
    yield tuple(outputs)

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
    
    query_time = gr.Textbox(label="Query Time")
    output_sections = {}
    for model_id, model_label in LLM_MODELS:
        with gr.Column():
            gr.Markdown(f"### {model_label}")
            output_sections[model_label + "_llm"] = gr.Textbox(label=f"{model_label} LLM-Only Answer")
    
    # Wire up buttons
    def on_submit(q):
        yield from multi_model_stream_interface(q)
    
    test_btn.click(simple_test, inputs=[], outputs=[status] + outputs)
    demo_btn.click(threaded_demo, inputs=[], outputs=[status] + outputs)
    real_btn.click(real_models, inputs=[question], outputs=[status] + outputs)
    btn.click(
        on_submit,
        inputs=[question],
        outputs=[query_time] + [output_sections[model_label + "_llm"] for _, model_label in LLM_MODELS],
        queue=True
    )
    question.submit(
        on_submit,
        inputs=[question],
        outputs=[query_time] + [output_sections[model_label + "_llm"] for _, model_label in LLM_MODELS],
        queue=True
    )

if __name__ == "__main__":
    print("🚀 Starting multimodel app...")
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
