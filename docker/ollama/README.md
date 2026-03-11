# Ollama

Standard [Ollama](https://ollama.com) LLM inference server. Run open-source language models locally with a simple API.

## Quick Start

```bash
cp .env.example .env
docker compose up -d
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

## Volumes

| Container Path | Description |
|---------------|-------------|
| `/root/.ollama` | Model weights, config, and cache |

## Notes

- Model storage can be large (7B models ~4GB, 70B models ~40GB). Point `OLLAMA_DATA` to a volume with sufficient space.
- To expose to LAN, ensure your firewall allows the configured port.
- For Intel GPU acceleration, see the [ollama-ipex](../ollama-ipex/) stack.
- For NVIDIA GPU support, add the NVIDIA Container Toolkit and `deploy.resources.reservations.devices` to the compose file.
