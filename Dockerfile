# Base image - Python 3.10 slim (smaller size)
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Environment variables for Python optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (for Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Install project as package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p logs artifacts

# Expose ports for API and Gradio
EXPOSE 8000

# Health check - verify API is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - run FastAPI app
CMD ["python", "app.py"]