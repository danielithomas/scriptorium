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
Two groups, both in the same `.env`. The first are read by **compose**; the rest
are passed **into the container** by `env_file: .env` and are what OpenCode
itself reads.

| Variable | Read by | Default | Description |
|----------|---------|---------|-------------|
| `WORKSPACE_PATH` | compose | `./workspace` | Host path mounted as the coding workspace |
| `OLLAMA_NETWORK` | compose | `ollama_default` | Existing Docker network shared with Ollama |
| `OLLAMA_HOST` | container | `http://ollama:11434` | Local Ollama endpoint, by container name on that network |
| `OLLAMA_CLOUD_URL` | container | *(none)* | Optional hosted endpoint for requests the local model cannot serve |
| `OLLAMA_API_KEY` | container | *(none)* | Key for `OLLAMA_CLOUD_URL`. Leave empty to stay entirely local |
| `LOCAL_MODEL` | container | *(none)* | Model to prefer locally |
| `FALLBACK_MODEL` | container | *(none)* | Model to fall back to |

**`LOCAL_MODEL` and `FALLBACK_MODEL` have no defaults and no fallback.** Without
them OpenCode starts normally and then has no model to select — it fails quietly
rather than loudly, which is the failure worth knowing about here.

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
