# Ollama — Intel IPEX-LLM (Arc / Meteor Lake iGPU)

[Ollama](https://ollama.com) with [IPEX-LLM](https://github.com/ipex-llm/ipex-llm) acceleration for Intel GPUs. Runs LLMs on Intel Arc discrete GPUs or Meteor Lake / Arrow Lake integrated GPUs via Level Zero.

## Prerequisites

- Intel Arc GPU or integrated GPU (Meteor Lake, Arrow Lake, etc.)
- Linux host with Intel GPU kernel driver installed
- `/dev/dri` device nodes accessible
- Docker with compose v2

### Verify GPU Access

```bash
# Check that render/video devices exist
ls -la /dev/dri/

# Verify Intel GPU is detected
sudo lspci | grep -i vga
```

## Quick Start

```bash
cp .env.example .env
# Edit .env if needed (port, data path, context size)
docker compose up -d --build
```

## Configuration

### Build Arguments

The Dockerfile uses build args to pin IPEX-LLM and driver versions. To update:

```bash
docker compose build --build-arg IPEXLLM_RELEASE_VERSION=v2.3.0-nightly \
  --build-arg IPEXLLM_PORTABLE_ZIP_FILENAME=ollama-ipex-llm-2.3.0b20250630-ubuntu.tgz
```

### Driver Version Matching

The Intel GPU user-space drivers in the Dockerfile **must match** your host kernel driver version. If you update your kernel driver, rebuild the container with matching versions from:

- [Level Zero releases](https://github.com/oneapi-src/level-zero/releases)
- [Intel Graphics Compiler releases](https://github.com/intel/intel-graphics-compiler/releases)
- [Compute Runtime releases](https://github.com/intel/compute-runtime/releases)

Check the compute-runtime release notes for compatible version sets.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_PORT` | `11435` | Host port (default differs from standard Ollama to allow coexistence) |
| `OLLAMA_DATA` | `./data` | Host path for model storage |
| `NUM_CTX` | `16384` | Default context window size |

## Running Alongside Standard Ollama

This stack defaults to port **11435** so it can run alongside a standard Ollama instance on 11434. Both can share the same model data directory.

```bash
# Standard Ollama (CPU)
curl http://localhost:11434/api/tags

# IPEX-LLM Ollama (Intel GPU)
curl http://localhost:11435/api/tags
```

## Shared Memory

The `shm_size: 16g` setting is required for Intel GPU inference. Reduce if your system has limited RAM, but performance may degrade.

## Troubleshooting

- **"No Intel GPU found"**: Ensure `/dev/dri/renderD128` exists and the container has device access
- **Slow first inference**: IPEX-LLM compiles GPU kernels on first run for each model; subsequent runs are faster
- **OOM errors**: Reduce `NUM_CTX` or use smaller models. iGPU shares system RAM.
- **Driver mismatch**: Rebuild with drivers matching your host kernel version
