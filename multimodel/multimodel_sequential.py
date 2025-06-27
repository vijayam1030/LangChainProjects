# Import Gradio for building the web app UI
import gradio as gr
# Import Ollama LLM from langchain_ollama (for LLM-only answers)
from langchain_ollama import OllamaLLM
import time
import os

# Get Ollama base URL from environment variable
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
print(f"Using Ollama base URL: {OLLAMA_BASE_URL}")

# Test Ollama connectivity at startup
try:
    import requests
    response = requests.get(f"{OLLAMA_BASE_URL}/api/version", timeout=10)
    print(f"Ollama connectivity test: {response.status_code}")
    print("Ollama is ready for connections")
except Exception as e:
    print(f"Ollama connectivity test failed: {e}")
    print("Continuing anyway - models will be tested when used...")

# List of available LLM models
LLM_MODELS = [
    ("llama2", "Llama 2"),
    ("qwen3:1.7b", "Qwen 3 1.7B"),
    ("gemma3:1b", "Gemma 3 1B"),
    ("deepseek-r1:1.5b", "DeepSeek R1 1.5B"),
    ("mistral:7b", "Mistral 7B"),
    ("phi3:3.8b", "Phi 3 3.8B"),
    ("tinyllama:1.1b", "TinyLlama 1.1B"),
    ("dolphin3:8b", "Dolphin 3 8B"),
    ("llama2-uncensored:7b", "Llama 2 U 7B"),
]

def run_model_sequentially(question, model_id, model_label):
    """Run a single model and return the result"""
    try:
        print(f"Processing {model_label}...")
        llm = OllamaLLM(
            model=model_id, 
            base_url=OLLAMA_BASE_URL,
            timeout=120
        )
        result = llm.invoke(question)
        print(f"Completed {model_label}")
        return result
    except Exception as e:
        error_msg = f"Error with {model_label}: {str(e)}"
        print(error_msg)
        return error_msg

def multi_model_sequential_interface(question, progress=gr.Progress(track_tqdm=True)):
    """Process all models sequentially instead of concurrently"""
    print(f"Starting sequential processing for: {question[:50]}...")
    start_time = time.time()
    
    results = {}
    
    # Process each model one at a time
    for i, (model_id, model_label) in enumerate(LLM_MODELS):
        if progress:
            progress(i / len(LLM_MODELS), desc=f"Processing {model_label}")
        
        result = run_model_sequentially(question, model_id, model_label)
        results[model_label] = result
        
        # Yield intermediate results
        elapsed = time.time() - start_time
        outputs = [f"⏱️ Query time: {elapsed:.2f} seconds (Processing {i+1}/{len(LLM_MODELS)})"]
        
        for _, label in LLM_MODELS:
            outputs.append(results.get(label, "Waiting..."))
        
        yield tuple(outputs)
    
    # Final result
    elapsed = time.time() - start_time
    outputs = [f"⏱️ Total time: {elapsed:.2f} seconds (Complete!)"]
    for _, label in LLM_MODELS:
        outputs.append(results.get(label, "Error"))
    
    yield tuple(outputs)

with gr.Blocks() as demo:
    gr.Markdown("# LLM Multi-Model (Sequential Version)\nThis app runs your question on multiple models sequentially to avoid connection issues.")
    
    with gr.Row():
        question = gr.Textbox(label="Enter your question:")
        btn = gr.Button("Submit", variant="primary")
    
    query_time = gr.Textbox(label="Status")
    
    output_sections = {}
    for model_id, model_label in LLM_MODELS:
        with gr.Column():
            gr.Markdown(f"### {model_label}")
            output_sections[model_label + "_llm"] = gr.Textbox(label=f"{model_label} Answer", lines=4)
    
    def on_submit(q):
        if not q.strip():
            return ["Please enter a question"] + [""] * len(LLM_MODELS)
        yield from multi_model_sequential_interface(q)
    
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

# Launch the application
def safe_launch():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} to launch Gradio interface...")
            demo.queue().launch(share=False, server_name="0.0.0.0", server_port=7860)
            break
        except Exception as e:
            print(f"Launch attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("All launch attempts failed")
                raise

if __name__ == "__main__":
    safe_launch()
