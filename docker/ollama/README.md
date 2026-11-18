# Ollama

Standard [Ollama](https://ollama.com) LLM inference server. Run open-source language models locally with a simple API.

## Quick Start

```bash
cp .env.example .env
docker compose up -d

# Intel iGPU / NPU, or any GPU through Vulkan, instead:
docker compose -f compose-igpu.yaml up -d

# AMD Radeon through ROCm instead:
docker compose -f compose-rocm.yaml up -d
```

## Pull & Run a Model

```bash
docker exec -it ollama ollama pull llama3.2
docker exec -it ollama ollama run llama3.2
```

## API

Ollama exposes an OpenAI-compatible API on port **11434**.

```bash
# List models
curl http://localhost:11434/api/tags

# Generate
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Hello, world!"
}'

# Chat (OpenAI-compatible)
curl http://localhost:11434/v1/chat/completions -d '{
  "model": "llama3.2",
  "messages": [{"role": "user", "content": "Hello"}]
}'
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_PORT` | `11434` | Host port for API |
| `OLLAMA_DATA` | `./data` | Host path for model storage |
| `OLLAMA_API_KEY` | *(none)* | API key for authenticating requests |
| `OLLAMA_BIND` | `0.0.0.0` | Host interface to publish on. `compose-igpu.yaml` and `compose-rocm.yaml` |
| `VIDEO_GID` / `RENDER_GID` | `44` / `992` | Group IDs passed to the container. `compose-igpu.yaml` and `compose-rocm.yaml` |
| `GGML_VK_VISIBLE_DEVICES` | `0` | Which Vulkan device to use, by index. With an iGPU and a discrete card, the iGPU is usually `0`. `compose-igpu.yaml` only |
| `OLLAMA_FLASH_ATTENTION` | *(Ollama's default)* | `1` turns flash attention on. `compose-rocm.yaml` only |
| `OLLAMA_KV_CACHE_TYPE` | *(Ollama's default, `f16`)* | `q8_0` halves the KV cache, and needs flash attention. `compose-rocm.yaml` only |

## Volumes

| Container Path | Description |
|---------------|-------------|
| `/root/.ollama` | Model weights, config, and cache |

## Notes

- Model storage can be large (7B models ~4GB, 70B models ~40GB). Point `OLLAMA_DATA` to a volume with sufficient space.
- To expose to LAN, ensure your firewall allows the configured port.
- **For Intel GPU acceleration, use `compose-igpu.yaml`** — it runs the upstream
  ollama image with its Vulkan backend enabled, passing through `/dev/dri` and
  (on Meteor Lake and later) the NPU at `/dev/accel/accel0`. Set `RENDER_GID`
  and `VIDEO_GID` from `getent group render video` first; they differ between
  distributions. The separate [ollama-ipex](../ollama-ipex/) stack is the older
  IPEX-based approach and needs a different image.
- **For an AMD Radeon, use `compose-rocm.yaml`.** It runs the image's `-rocm`
  build, passing through `/dev/kfd` and `/dev/dri`. The host needs only the
  kernel's `amdgpu` driver, because the image carries the ROCm runtime.
  `compose-igpu.yaml` can drive a Radeon too, through Vulkan: set
  `GGML_VK_VISIBLE_DEVICES` to its index. Measured on an RX 9070 XT with
  Ollama 0.40.13, ROCm read prompts about twice as fast as Vulkan, and
  generated at about the same speed. Embeddings were a quarter slower.
- **A headless Radeon may need `amdgpu.runpm=0`** on hosts whose firmware
  offers a hybrid-graphics power switch (ACPI `ATPX`) without the methods it
  calls. There, the driver powers the idle card down, and it fails to come back
  ("resume of IP block <smu> failed"). Set `options amdgpu runpm=0` in
  `/etc/modprobe.d/`, then rebuild the initramfs.
- For NVIDIA GPU support, add the NVIDIA Container Toolkit and `deploy.resources.reservations.devices` to the compose file.
