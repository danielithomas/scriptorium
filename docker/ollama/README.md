# Ollama

Standard [Ollama](https://ollama.com) LLM inference server. Run open-source language models locally with a simple API.

## Quick Start

```bash
cp .env.example .env
docker compose up -d

# Intel iGPU / NPU instead:
docker compose -f compose-igpu.yaml up -d
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
| `OLLAMA_BIND` | `0.0.0.0` | Host interface to publish on. `compose-igpu.yaml` only |
| `VIDEO_GID` / `RENDER_GID` | `44` / `992` | Group IDs passed to the container. `compose-igpu.yaml` only |
| `GGML_VK_VISIBLE_DEVICES` | `0` | Which Vulkan device to use. `compose-igpu.yaml` only |

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
- For NVIDIA GPU support, add the NVIDIA Container Toolkit and `deploy.resources.reservations.devices` to the compose file.
