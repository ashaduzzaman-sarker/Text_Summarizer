# ============================================================================
# Dockerfile
# ============================================================================
# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy trained model (or download during build)
# COPY artifacts/model_trainer/final_model ./artifacts/model_trainer/final_model

# Expose ports
EXPOSE 8000 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=artifacts/model_trainer/final_model

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python", "app.py"]