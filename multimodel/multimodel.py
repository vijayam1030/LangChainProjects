# Import Gradio for building the web app UI
import gradio as gr
# Import Ollama LLM from langchain_ollama (for LLM-only answers)
from langchain_ollama import OllamaLLM
import time
import threading
import queue

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
    ("llama2-uncensored:7b", "Llama 2  7B"),
]

def llm_only_answer(question, llm_model):
    llm = OllamaLLM(model=llm_model)
    return llm.invoke(question)

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
        finished = sum([t.done() if hasattr(t, 'done') else not t.is_alive() for t in threads])
    # Final yield to ensure all answers are shown
    elapsed = time.time() - start_time
    outputs = [f"⏱️ Query time: {elapsed:.2f} seconds"]
    for _, model_label in LLM_MODELS:
        outputs.append(answers[model_label])
    yield tuple(outputs)

with gr.Blocks() as demo:
    gr.Markdown("# LLM Multi-Model \nThis app runs your question on multiple models in parallel and shows LLM-only answers for each.")
    with gr.Row():
        question = gr.Textbox(label="Enter your question:")
        btn = gr.Button("Submit", variant="primary")
    query_time = gr.Textbox(label="Query Time")
    output_sections = {}
    for model_id, model_label in LLM_MODELS:
        with gr.Column():
            gr.Markdown(f"### {model_label}")
            output_sections[model_label + "_llm"] = gr.Textbox(label=f"{model_label} LLM-Only Answer")
    def on_submit(q):
        yield from multi_model_stream_interface(q)
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

demo.queue().launch(share=True)
