FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ARG COMFYUI_VERSION=master

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create venv to avoid PEP 668 issues
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA 12.8 support (required for Blackwell / RTX 50-series / sm_120)
# Using nightly until stable PyTorch includes sm_120 kernels
RUN pip install --no-cache-dir --pre \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Clone ComfyUI
RUN git clone --depth 1 --branch ${COMFYUI_VERSION} \
    https://github.com/comfyanonymous/ComfyUI.git /app

WORKDIR /app

# Install ComfyUI dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages for FLUX support
RUN pip install --no-cache-dir \
    accelerate \
    sentencepiece \
    protobuf

EXPOSE 8188

# --listen: bind all interfaces (required inside Docker)
# --output-directory: configurable via volume mount
CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
