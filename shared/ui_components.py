# Shared UI components for Gradio applications

import gradio as gr
from typing import List, Tuple, Dict, Any

class GradioComponents:
    """Reusable Gradio UI components"""
    
    @staticmethod
    def create_model_selector(models: List[Tuple[str, str]], label: str = "Select Model") -> gr.Dropdown:
        """Create a model selection dropdown"""
        return gr.Dropdown(
            choices=[(label, model_id) for model_id, label in models],
            label=label,
            value=models[0][0] if models else None
        )
    
    @staticmethod
    def create_question_input(placeholder: str = "Enter your question...") -> gr.Textbox:
        """Create question input textbox"""
        return gr.Textbox(
            label="Question",
            placeholder=placeholder,
            lines=2
        )
    
    @staticmethod
    def create_answer_output(label: str = "Answer") -> gr.Textbox:
        """Create answer output textbox"""
        return gr.Textbox(
            label=label,
            lines=10,
            max_lines=20,
            show_copy_button=True
        )
    
    @staticmethod
    def create_time_display(label: str = "Query Time") -> gr.Textbox:
        """Create time display textbox"""
        return gr.Textbox(
            label=label,
            interactive=False
        )

class UIThemes:
    """Predefined UI themes and styles"""
    
    DARK_THEME = gr.themes.Soft(
        primary_hue="slate",
        secondary_hue="gray",
        neutral_hue="slate"
    )
    
    LIGHT_THEME = gr.themes.Default(
        primary_hue="blue",
        secondary_hue="gray"
    )

def create_header(title: str, description: str = "") -> str:
    """Create markdown header for apps"""
    header = f"# {title}"
    if description:
        header += f"\n{description}"
    return header

def create_footer() -> str:
    """Create standard footer"""
    return "---\n*Powered by LangChain and Ollama*"
