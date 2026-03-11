FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

# Base packages
RUN apt update && \
    apt install --no-install-recommends -q -y \
    software-properties-common \
    ca-certificates \
    wget \
    ocl-icd-libopencl1

# Intel GPU compute user-space drivers
# Download matching versions from:
#   - https://github.com/oneapi-src/level-zero/releases
#   - https://github.com/intel/intel-graphics-compiler/releases
#   - https://github.com/intel/compute-runtime/releases
#
# These versions must be compatible with each other and your kernel driver.
# Check Intel's compute-runtime release notes for compatible sets.
RUN mkdir -p /tmp/gpu && \
    cd /tmp/gpu && \
    wget https://github.com/oneapi-src/level-zero/releases/download/v1.21.9/level-zero_1.21.9+u24.04_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.8.3/intel-igc-core-2_2.8.3+18762_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.8.3/intel-igc-opencl-2_2.8.3+18762_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/25.09.32961.7/intel-level-zero-gpu_1.6.32961.7_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/25.09.32961.7/intel-opencl-icd_25.09.32961.7_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/25.09.32961.7/libigdgmm12_22.6.0_amd64.deb && \
    dpkg -i *.deb && \
    rm -rf /tmp/gpu

# Install IPEX-LLM Ollama portable binary
# See: https://github.com/ipex-llm/ipex-llm/releases
ARG IPEXLLM_RELEASE_REPO=ipex-llm/ipex-llm
ARG IPEXLLM_RELEASE_VERSION=v2.3.0-nightly
ARG IPEXLLM_PORTABLE_ZIP_FILENAME=ollama-ipex-llm-2.3.0b20250630-ubuntu.tgz
RUN cd / && \
    wget https://github.com/${IPEXLLM_RELEASE_REPO}/releases/download/${IPEXLLM_RELEASE_VERSION}/${IPEXLLM_PORTABLE_ZIP_FILENAME} && \
    tar xvf ${IPEXLLM_PORTABLE_ZIP_FILENAME} --strip-components=1 -C / && \
    rm ${IPEXLLM_PORTABLE_ZIP_FILENAME}

# Fix start script to respect OLLAMA_HOST env var instead of hardcoding localhost
RUN sed -i "s|export OLLAMA_HOST='127.0.0.1:11434'|export OLLAMA_HOST=\${OLLAMA_HOST:-'0.0.0.0:11434'}|" /start-ollama.sh

ENV OLLAMA_HOST=0.0.0.0:11434

ENTRYPOINT ["/bin/bash", "/start-ollama.sh"]
