# Use official PyTorch image with CUDA 12.8 for Blackwell (B200) support
# Also compatible with Hopper (H100/H200) architecture
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# Cache bust - change this value to force rebuild without cache
ARG CACHE_VERSION=2025-01-18-v1
RUN echo "Cache version: $CACHE_VERSION"

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY handler.py /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run the handler
CMD ["python", "handler.py"]
