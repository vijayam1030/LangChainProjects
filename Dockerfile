# Multi-stage Dockerfile for LangChain applications
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set Python path to include shared modules
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose ports for different apps
EXPOSE 7860 8501 8000

# Default environment variables
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV STREAMLIT_SERVER_PORT=8501
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

# Default command (can be overridden)
CMD ["python", "multimodel/multimodel.py"]
