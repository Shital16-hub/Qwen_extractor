FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gradio

# Create necessary directories
RUN mkdir -p data/samples models uploads

# Copy all source code and files
COPY src /app/src
COPY .gitignore /app/.gitignore
COPY README.md /app/README.md

# Expose port for the web interface
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command
CMD ["python", "src/app.py"]