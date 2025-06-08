# Use official Python image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY ../requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# Copy the multimodel app code
COPY . .

# Pull llama2 model before running the app
RUN pip install --no-cache-dir ollama && \
    ollama pull llama2 || true

# Expose Gradio default port
EXPOSE 7860

# Set environment variable to allow Gradio to be accessed from anywhere
ENV GRADIO_SERVER_NAME=0.0.0.0

# Run the multimodel app
CMD ["python", "multimodel.py"]
