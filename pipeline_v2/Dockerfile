FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# PyTorch will be upgraded to nightly for RTX 5070 Ti (Blackwell, sm_120) support
# Base image provides PyTorch 2.5.1, but requirements.txt will install nightly build
# Clean PyTorch + CUDA environment optimized for RTX 5070 Ti (16GB)
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_ALLOW_TF32_CUBLAS=1 \
    CUDA_MODULE_LOADING=LAZY \
    TORCH_CUDNN_V8_API_ENABLED=1

# Install system dependencies for CV/ML workloads
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
      build-essential g++ gcc \
      libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
      libgl1-mesa-glx libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies (layer cached for faster rebuilds)
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt && \
    # Force upgrade PyTorch to nightly for RTX 5070 Ti (sm_120) support
    # This will also upgrade all CUDA libraries to compatible versions
    pip install --upgrade --force-reinstall \
        --extra-index-url https://download.pytorch.org/whl/nightly/cu124 \
        torch torchvision torchaudio

# Copy project code
COPY . /workspace

# Keep container running for interactive CLI access
CMD ["tail", "-f", "/dev/null"]
