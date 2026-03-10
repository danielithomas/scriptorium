FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create venv to avoid PEP 668 issues
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA 12.8 support (required for Blackwell / RTX 50-series / sm_120)
# Using nightly until stable PyTorch includes sm_120 kernels
RUN pip install --no-cache-dir --pre \
    torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

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
