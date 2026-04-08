FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# ── Environment — RTX 5070 Ti (Blackwell, sm_120) optimized ──────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_ALLOW_TF32_CUBLAS=1 \
    CUDA_MODULE_LOADING=LAZY \
    TORCH_CUDNN_V8_API_ENABLED=1

# ── System deps (OpenCV, OpenGL, git for torch.hub) ─────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
      build-essential g++ gcc \
      libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
      libgl1-mesa-glx libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── CUDA 13.0 NVRTC builtins — required by PyTorch nightly (cu130) ──────────
# The nightly torch wheel links against libnvrtc-builtins.so.13.0 for NVRTC
# JIT compilation (fused kernels). The base image only ships 12.4, so we add
# the CUDA 13.0 repository and install just the nvrtc runtime packages.
RUN curl -fsSL \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
      -o /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
         cuda-nvrtc-13-0 \
         cuda-nvrtc-dev-13-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps (layer-cached for faster rebuilds) ──────────────────────────
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt && \
    # termcolor is required by the DINOv3 backbone (pulled via torch.hub)
    pip install termcolor && \
    # Force nightly PyTorch for RTX 5070 Ti (sm_120) support.
    # cu130 index is explicit — the cu124 index now silently serves cu130 builds
    # which mismatches the 12.4 NVRTC builtins and causes kernel JIT failures.
    pip install --upgrade --force-reinstall \
        --extra-index-url https://download.pytorch.org/whl/nightly/cu130 \
        torch torchvision torchaudio

# ── Copy project code ────────────────────────────────────────────────────────
COPY . /app

ENV PYTHONPATH=/app:/app/core \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

EXPOSE 8000

# ── Run the FastAPI service ──────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
