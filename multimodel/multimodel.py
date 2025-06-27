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

def multi_model_sequential_interface(question):
    """Process all models sequentially with real-time streaming results"""
    print(f"🚀 STREAMING: Starting sequential processing for: {question[:50]}...")
    start_time = time.time()
    
    results = {}
    
    # Initialize all outputs with "⏳ Waiting..." status
    initial_outputs = [f"🚀 Starting processing... (0/{len(LLM_MODELS)} complete)"]
    for _, label in LLM_MODELS:
        initial_outputs.append("⏳ Waiting...")
    
    print(f"🔄 STREAMING: Yielding initial state")
    yield initial_outputs
    time.sleep(0.5)  # Small delay to make streaming visible
    
    # Process each model one at a time and stream results immediately
    for i, (model_id, model_label) in enumerate(LLM_MODELS):
        # Update status to show current model being processed
        processing_outputs = [f"🔄 Processing {model_label}... ({i}/{len(LLM_MODELS)} complete)"]
        for _, label in LLM_MODELS:
            if label == model_label:
                processing_outputs.append("🔄 Processing...")
            else:
                processing_outputs.append(results.get(label, "⏳ Waiting..."))
        
        print(f"🔄 STREAMING: Yielding processing state for {model_label}")
        yield processing_outputs
        time.sleep(0.3)  # Small delay to make streaming visible
        
        # Run the model
        print(f"🤖 PROCESSING: Running {model_label}...")
        result = run_model_sequentially(question, model_id, model_label)
        results[model_label] = result
        print(f"✅ COMPLETED: {model_label} finished")
        
        # Stream the result immediately after completion
        elapsed = time.time() - start_time
        outputs = [f"⏱️ Query time: {elapsed:.2f} seconds ({i+1}/{len(LLM_MODELS)} complete)"]
        
        for _, label in LLM_MODELS:
            if label in results:
                outputs.append(f"✅ {results[label]}")
            else:
                outputs.append("⏳ Waiting...")
        
        print(f"✅ STREAMING: Yielding result for {model_label}")
        yield outputs
        time.sleep(0.2)  # Small delay before next model
    
    # Final result with completion status
    elapsed = time.time() - start_time
    outputs = [f"🎉 All models complete! Total time: {elapsed:.2f} seconds"]
    for _, label in LLM_MODELS:
        outputs.append(f"✅ {results.get(label, '❌ Error')}")
    
    print(f"🎉 STREAMING: Yielding final results")
    yield outputs

with gr.Blocks() as demo:
    gr.Markdown("# 🚀 LLM Multi-Model (Real-time Streaming)\nThis app processes your question across multiple models **sequentially** and streams results in real-time as each model completes. No need to wait for everything to finish!")
    
    with gr.Row():
        question = gr.Textbox(label="Enter your question:", placeholder="Ask anything... results will stream as each model completes!")
        btn = gr.Button("🚀 Submit", variant="primary")
    
    with gr.Row():
        demo_btn = gr.Button("🎭 Demo Mode (Test Streaming)", variant="secondary")
    
    query_time = gr.Textbox(label="Status")
    
    output_sections = []
    for model_id, model_label in LLM_MODELS:
        with gr.Column():
            gr.Markdown(f"### {model_label}")
            output_sections.append(gr.Textbox(label=f"{model_label} Answer", lines=4))
    
    def demo_mode():
        """Demo mode with fake responses to test streaming"""
        print("🎭 DEMO MODE: Starting fake streaming demo...")
        
        # Initialize
        outputs = ["🎭 Demo Mode: Testing streaming..."] + ["⏳ Waiting..."] * len(LLM_MODELS)
        yield outputs
        time.sleep(1)
        
        # Process each "model" quickly
        for i, (_, model_label) in enumerate(LLM_MODELS):
            # Processing state
            outputs = [f"🔄 Demo: Processing {model_label}... ({i}/{len(LLM_MODELS)} complete)"]
            for j, (_, label) in enumerate(LLM_MODELS):
                if j < i:
                    outputs.append(f"✅ Demo response from {label}: This is a test response!")
                elif j == i:
                    outputs.append("🔄 Processing...")
                else:
                    outputs.append("⏳ Waiting...")
            yield outputs
            time.sleep(1)
            
            # Completed state
            outputs = [f"⏱️ Demo: Completed {i+1}/{len(LLM_MODELS)} models"]
            for j, (_, label) in enumerate(LLM_MODELS):
                if j <= i:
                    outputs.append(f"✅ Demo response from {label}: This is a test response!")
                else:
                    outputs.append("⏳ Waiting...")
            yield outputs
            time.sleep(0.5)
        
        # Final
        outputs = ["🎉 Demo Complete! Streaming works!"] + [f"✅ Demo response from {label}: This is a test response!" for _, label in LLM_MODELS]
        yield outputs
    
    def on_submit(q):
        if not q.strip():
            error_outputs = ["Please enter a question"] + [""] * len(LLM_MODELS)
            yield error_outputs
            return
        
        # Return the generator directly for streaming
        for outputs in multi_model_sequential_interface(q):
            yield outputs
    
    btn.click(
        on_submit,
        inputs=[question],
        outputs=[query_time] + output_sections
    )
    
    demo_btn.click(
        demo_mode,
        inputs=[],
        outputs=[query_time] + output_sections
    )
    
    question.submit(
        on_submit,
        inputs=[question],
        outputs=[query_time] + output_sections
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
