#!/usr/bin/env python3
"""
Minimal streaming test to isolate the issue
"""
import gradio as gr
import time

def simple_streaming_test():
    """Simple test that should definitely work"""
    for i in range(5):
        yield f"Update {i+1}: Streaming works! Time: {time.time():.1f}"
        time.sleep(1)
    yield "Final: Streaming completed successfully!"

def simple_multiple_outputs():
    """Test with multiple outputs like our real app"""
    for i in range(3):
        outputs = [
            f"Status: Update {i+1}",
            f"Output 1: Result {i+1}",
            f"Output 2: Result {i+1}",
            f"Output 3: Result {i+1}"
        ]
        yield outputs
        time.sleep(1)
    
    final_outputs = [
        "Status: All complete!",
        "Output 1: Final result 1",
        "Output 2: Final result 2", 
        "Output 3: Final result 3"
    ]
    yield final_outputs

with gr.Blocks() as demo:
    gr.Markdown("# 🧪 Streaming Test")
    
    with gr.Row():
        test1_btn = gr.Button("Test 1: Simple Streaming")
        test2_btn = gr.Button("Test 2: Multiple Outputs")
    
    # Test 1: Single output
    output1 = gr.Textbox(label="Single Output Test")
    
    # Test 2: Multiple outputs (like our real app)
    status = gr.Textbox(label="Status")
    out1 = gr.Textbox(label="Output 1")
    out2 = gr.Textbox(label="Output 2") 
    out3 = gr.Textbox(label="Output 3")
    
    test1_btn.click(
        simple_streaming_test,
        inputs=[],
        outputs=[output1]
    )
    
    test2_btn.click(
        simple_multiple_outputs,
        inputs=[],
        outputs=[status, out1, out2, out3]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
