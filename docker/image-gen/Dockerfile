FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ocl-icd-libopencl1 \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] \
    optimum[openvino] \
    openvino \
    diffusers transformers accelerate \
    torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu \
    pillow \
    python-multipart \
    realesrgan basicsr gfpgan

# Install Intel GPU drivers AFTER pip install (to leverage cache)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gpg \
    && wget -qO- https://repositories.intel.com/gpu/intel-graphics.key | gpg --dearmor > /usr/share/keyrings/intel-graphics.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble unified" > /etc/apt/sources.list.d/intel-gpu.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends intel-level-zero-gpu level-zero \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY server.py .

EXPOSE 8100

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8100"]
