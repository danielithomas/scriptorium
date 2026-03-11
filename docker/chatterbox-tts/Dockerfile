FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ARG RUNTIME=nvidia

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/app/hf_cache

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install Python dependencies (CPU by default)
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Conditionally install NVIDIA GPU dependencies
COPY requirements-nvidia.txt .
RUN if [ "$RUNTIME" = "nvidia" ]; then \
    pip3 install --no-cache-dir -r requirements-nvidia.txt; \
    fi

COPY . .

RUN mkdir -p model_cache reference_audio outputs voices logs hf_cache

EXPOSE 8004

CMD ["python3", "server.py"]
