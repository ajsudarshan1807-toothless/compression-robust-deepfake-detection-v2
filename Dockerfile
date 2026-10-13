# Dockerfile for Compression-Robust Deepfake Detection Service
FROM python:3.12-slim

# System dependencies for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package specifications
COPY pyproject.toml README.md ./

# Install python package and dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy application code and configuration
COPY deepfake_detector/ deepfake_detector/
COPY apps/ apps/
COPY configs/ configs/
COPY docs/ docs/

# Expose API port (FastAPI) and Streamlit fallback port
EXPOSE 8000 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Default command: start FastAPI service
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
