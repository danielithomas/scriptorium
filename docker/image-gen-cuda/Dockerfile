FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
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
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install diffusers stack
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] \
    diffusers transformers accelerate \
    safetensors \
    pillow \
    python-multipart \
    realesrgan basicsr gfpgan \
    xformers

WORKDIR /app
COPY server.py .

EXPOSE 8100

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8100"]
