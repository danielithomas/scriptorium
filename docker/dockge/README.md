# Dockge

[Dockge](https://github.com/louislam/dockge) — a Docker Compose stack manager with a clean web UI. Create, edit, start, stop, and monitor your Docker Compose stacks from a browser.

## Quick Start

```bash
cp .env.example .env
# Edit STACKS_DIR to point to your Docker Compose stacks directory

docker compose up -d
```

Access the UI at `http://localhost:5001`.

## How It Works

Dockge scans the `STACKS_DIR` directory for Docker Compose files. Each subdirectory containing a `compose.yaml` or `docker-compose.yml` becomes a manageable stack in the UI.

```
/data/docker/              ← STACKS_DIR
├── my-app/
│   └── compose.yaml
├── monitoring/
│   └── compose.yaml
└── ...
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCKGE_PORT` | `5001` | Host port for web UI |
| `STACKS_DIR` | `/data/docker` | Host path to Docker Compose stacks directory |

## Security Notes

- **Docker socket access**: Dockge mounts `/var/run/docker.sock`, giving it full Docker control. Only expose on trusted networks.
- **No built-in auth**: Consider placing behind a reverse proxy with authentication for production use.
- **LAN only**: Do not expose Dockge to the internet without additional security measures.

## Notes

- Dockge manages Compose stacks, not individual containers.
- It reads and writes `compose.yaml` files directly — changes in the UI are reflected in the files and vice versa.
- Stack names are derived from directory names.
