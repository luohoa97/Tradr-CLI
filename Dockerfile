FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
# Add safetensors, huggingface_hub, scikit-learn explicitly just in case
RUN pip install --no-cache-dir -r requirements.txt \
    safetensors huggingface_hub scikit-learn pandas numpy torch yfinance

# Copy project files
COPY . .

# Environment variables (to be set in HF Space Secrets)
ENV HF_HOME=/tmp/huggingface
ENV HF_REPO_ID=""
ENV HF_TOKEN=""

# Command to run training
# This will output the performance report and upload to HF Hub
CMD ["python", "scripts/train_ai_model.py"]
