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
#
# STABLE cu128, not nightly: the nightly index rebuilds torch, torchvision and
# torchaudio independently, so a set pulled on different days fails to resolve
# against each other (ResolutionImpossible). Stable has shipped sm_120 since 2.7.
RUN pip install --no-cache-dir --retries 5 --timeout 120 \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

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

# Install ComfyUI-GGUF extension (enables GGUF quantised model loading)
# Provides UnetLoaderGGUF, CLIPLoaderGGUF, etc. for running Q4/Q5/Q8 models
RUN git clone --depth 1 \
    https://github.com/city96/ComfyUI-GGUF.git /app/custom_nodes/ComfyUI-GGUF \
    && pip install --no-cache-dir -r /app/custom_nodes/ComfyUI-GGUF/requirements.txt 2>/dev/null || true

EXPOSE 8188

# --listen: bind all interfaces (required inside Docker)
# --output-directory: configurable via volume mount
CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
