# OpenCode

[OpenCode](https://opencode.ai) AI coding assistant running in a Docker container, connected to a local Ollama instance for fully offline AI-assisted development.

## Prerequisites

- A running Ollama instance (see [ollama](../ollama/) stack)
- The Ollama Docker network must exist before starting this container

## Quick Start

```bash
cp .env.example .env

# Ensure the Ollama network exists
docker network ls | grep ollama_default

# Create workspace directory
mkdir -p workspace

# Build and start
docker compose up -d --build
```

## Usage

Attach to the container for an interactive session:

```bash
docker attach opencode

# Inside the container:
opencode
```

Detach with `Ctrl+P, Ctrl+Q` (keeps the container running).

## Network Setup

OpenCode connects to Ollama via a shared Docker network. The default assumes an `ollama_default` network created by the Ollama compose stack.

If your Ollama network has a different name:

```bash
# Check existing networks
docker network ls | grep ollama

# Update OLLAMA_NETWORK in .env
OLLAMA_NETWORK=my_ollama_network
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKSPACE_PATH` | `./workspace` | Host path mounted as the coding workspace |
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama API endpoint (container name on shared network) |
| `OLLAMA_NETWORK` | `ollama_default` | Docker network shared with Ollama |

## Volumes

| Path | Description |
|------|-------------|
| `/home/coder/workspace` | Mounted workspace for project files |
| `opencode-config` | Persistent OpenCode configuration (named volume) |

## Notes

- The container runs as a non-root `coder` user with passwordless sudo.
- OpenCode config persists across container rebuilds via the named volume.
- The workspace is bind-mounted — files edited inside the container are immediately available on the host and vice versa.
- For cloud LLM providers instead of Ollama, set the appropriate API key environment variables in the compose file.
