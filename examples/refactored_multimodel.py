# Example of how to refactor your existing multimodel.py using shared utilities

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# Import shared utilities
from shared.llm_utils import LLMManager, format_query_time
from shared.ui_components import GradioComponents, create_header
from config.settings import ModelConfig, AppConfig

def run_llm_stream_for_model(question: str, model_id: str, out_queue: queue.Queue, label: str):
    """Stream LLM response for a single model"""
    try:
        llm = LLMManager.create_llm(model_id, streaming=True)
        answer = ""
        for chunk in llm.stream(question):
            answer += chunk
            out_queue.put((label, answer))
    except Exception as e:
        out_queue.put((label, f"Error: {str(e)}"))

def multi_model_stream_interface(question: str, progress=gr.Progress(track_tqdm=True)):
    """Run question on multiple models with streaming"""
    out_queue = queue.Queue()
    threads = []
    
    # Get models from config
    models = ModelConfig.get_model_list()
    
    # Start threads for each model
    for model_id, model_label in models:
        t = threading.Thread(
            target=run_llm_stream_for_model, 
            args=(question, model_id, out_queue, model_label)
        )
        t.start()
        threads.append(t)
    
    answers = {label: "" for _, label in models}
    finished = 0
    total = len(models)
    start_time = time.time()
    
    while finished < total:
        try:
            label, ans = out_queue.get(timeout=AppConfig.STREAMING_TIMEOUT)
            answers[label] = ans
            elapsed = time.time() - start_time
            
            # Prepare outputs
            outputs = [format_query_time(elapsed)]
            for _, model_label in models:
                outputs.append(answers[model_label])
            
            yield tuple(outputs)
        except queue.Empty:
            pass
        
        finished = sum([not t.is_alive() for t in threads])
    
    # Final yield
    elapsed = time.time() - start_time
    outputs = [format_query_time(elapsed)]
    for _, model_label in models:
        outputs.append(answers[model_label])
    yield tuple(outputs)

def create_app():
    """Create the Gradio app using shared components"""
    models = ModelConfig.get_model_list()
    
    with gr.Blocks(title="Multi-Model LLM Comparison") as demo:
        gr.Markdown(create_header(
            "🤖 Multi-Model LLM Comparison",
            "Compare responses from multiple language models in parallel"
        ))
        
        with gr.Row():
            question = GradioComponents.create_question_input(
                "Ask a question to compare across all models..."
            )
            btn = gr.Button("🚀 Submit", variant="primary")
        
        query_time = GradioComponents.create_time_display()
        
        # Create output sections for each model
        output_sections = {}
        for model_id, model_label in models:
            with gr.Column():
                gr.Markdown(f"### {model_label}")
                output_sections[model_label] = GradioComponents.create_answer_output(
                    f"{model_label} Response"
                )
        
        def on_submit(q):
            yield from multi_model_stream_interface(q)
        
        # Wire up the interface
        btn.click(
            on_submit,
            inputs=[question],
            outputs=[query_time] + [output_sections[label] for _, label in models],
            queue=True
        )
        
        question.submit(
            on_submit,
            inputs=[question],
            outputs=[query_time] + [output_sections[label] for _, label in models],
            queue=True
        )
    
    return demo

if __name__ == "__main__":
    app = create_app()
    app.queue().launch(
        share=AppConfig.GRADIO_SHARE,
        server_name=AppConfig.GRADIO_SERVER_NAME,
        server_port=AppConfig.GRADIO_SERVER_PORT
    )
