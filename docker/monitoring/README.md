# Monitoring — Uptime Kuma + Glances

Combined monitoring stack: [Uptime Kuma](https://github.com/louislam/uptime-kuma) for service availability monitoring and [Glances](https://github.com/nicolargo/glances) for real-time system metrics.

## Quick Start

```bash
cp .env.example .env
mkdir -p uptime-kuma-data
docker compose up -d
```

## Services

### Uptime Kuma — Service Monitor

Access at `http://localhost:3001`

- Monitor HTTP/HTTPS endpoints, TCP ports, DNS, Docker containers, and more
- Notifications via email, Slack, Telegram, Discord, webhooks, etc.
- Status pages for public or internal dashboards
- First-time setup prompts for admin account creation

### Glances — System Metrics

Access at `http://localhost:61208`

- Real-time CPU, memory, disk, network, and process monitoring
- Docker container stats (via docker.sock)
- Web UI with auto-refresh
- REST API for programmatic access

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KUMA_PORT` | `3001` | Host port for Uptime Kuma UI |
| `KUMA_DATA` | `./uptime-kuma-data` | Host path for Uptime Kuma data |
| `GLANCES_PORT` | `61208` | Host port for Glances web UI |

## Glances API

```bash
# System overview
curl http://localhost:61208/api/4/all

# CPU usage
curl http://localhost:61208/api/4/cpu

# Memory
curl http://localhost:61208/api/4/mem

# Docker containers
curl http://localhost:61208/api/4/containers
```

## Security Notes

- **Docker socket**: Glances mounts `/var/run/docker.sock` read-only for container monitoring. Only expose on trusted networks.
- **PID namespace**: Glances uses `pid: host` to see all host processes. This is required for accurate process monitoring.
- **Uptime Kuma**: Has built-in authentication. Set a strong password on first setup.

## Running Separately

If you only need one of the services, you can start them individually:

```bash
docker compose up -d uptime-kuma
# or
docker compose up -d glances
```

## Notes

- Uptime Kuma data includes monitor configurations and history — back up the data directory regularly.
- Glances is stateless — no persistent data needed.
- Both services are lightweight and suitable for always-on monitoring on low-power hardware.
