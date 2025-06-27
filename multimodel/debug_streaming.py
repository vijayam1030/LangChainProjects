#!/usr/bin/env python3
"""
Minimal test to diagnose the streaming issue
"""
import gradio as gr
import time

def simple_test():
    """Basic streaming test without any LLM calls"""
    for i in range(5):
        yield f"Test {i+1}: Basic streaming works! Time: {time.time():.2f}"
        time.sleep(0.5)
    yield "✅ Basic streaming test completed!"

def fake_llm_test():
    """Test with fake LLM responses to see if the UI updates"""
    models = ["TinyLlama", "Qwen3", "Gemma3"]
    results = [""] * len(models)
    
    for step in range(10):
        # Simulate one model completing
        if step == 3:
            results[0] = "TinyLlama: 1+1=2 (this is a fake response)"
        elif step == 6:
            results[1] = "Qwen3: 1+1 equals 2 (this is also fake)"
        elif step == 9:
            results[2] = "Gemma3: The answer is 2 (fake response)"
        
        # Build outputs
        status = f"Step {step+1}/10 - {sum(1 for r in results if r)}/3 models complete"
        outputs = [status] + [r if r else f"🔄 Processing..." for r in results]
        yield outputs
        time.sleep(0.5)
    
    final = ["✅ Fake LLM test completed!"] + results
    yield final

# Minimal Gradio interface
with gr.Blocks(title="Debug Streaming") as app:
    gr.Markdown("# 🐛 Debug Streaming Test")
    
    with gr.Row():
        test1_btn = gr.Button("Test 1: Basic Streaming")
        test2_btn = gr.Button("Test 2: Fake LLM Streaming")
    
    output1 = gr.Textbox(label="Basic Test Output")
    
    status = gr.Textbox(label="Status")
    model1 = gr.Textbox(label="Model 1")
    model2 = gr.Textbox(label="Model 2") 
    model3 = gr.Textbox(label="Model 3")
    
    test1_btn.click(simple_test, outputs=[output1])
    test2_btn.click(fake_llm_test, outputs=[status, model1, model2, model3])

if __name__ == "__main__":
    print("🐛 Starting debug app on port 7860...")
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
