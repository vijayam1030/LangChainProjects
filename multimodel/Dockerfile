# Use official Python image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY ../requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# Copy the multimodel app code
COPY . .

# Note: Models will be pulled at runtime to avoid Docker build timeouts
# The models used are: tinyllama:1.1b, qwen3:1.7b, gemma3:1b

# Expose Gradio default port
EXPOSE 7860

# Set environment variable to allow Gradio to be accessed from anywhere
ENV GRADIO_SERVER_NAME=0.0.0.0

# Run the multimodel app
CMD ["python", "multimodel.py"]
