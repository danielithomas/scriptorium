# Caddy — LAN Reverse Proxy

[Caddy](https://caddyserver.com) as a lightweight reverse proxy for LAN Docker services. Maps multiple internal services to friendly ports with minimal configuration.

## Quick Start

```bash
# Edit the Caddyfile to add your services
vim Caddyfile

docker compose up -d
```

Verify it's running: `curl http://localhost:80`

## Configuration

Edit `Caddyfile` to add reverse proxy rules. Each block maps an incoming port to an upstream service:

```
# Proxy requests on :8080 to a container named "myapp" on port 3000
:8080 {
    reverse_proxy myapp:3000
}

# Proxy to a service running on the Docker host (not in a container)
:8081 {
    reverse_proxy host.docker.internal:9090
}
```

After editing, reload the config:

```bash
docker exec caddy caddy reload --config /etc/caddy/Caddyfile
```

## Proxying to Host Services

The `extra_hosts` mapping in `compose.yaml` makes `host.docker.internal` resolve to the Docker host IP. Use this to proxy services that run directly on the host (not in containers).

## HTTPS / Automatic TLS

For LAN use with IP addresses or `.local` names, Caddy serves plain HTTP. For automatic HTTPS:

1. Use a real domain name (e.g., `myservice.example.com`)
2. Ensure ports 80 and 443 are accessible
3. Caddy will automatically obtain and renew Let's Encrypt certificates

## Volumes

| Volume | Description |
|--------|-------------|
| `caddy_data` | TLS certificates and persistent data |
| `caddy_config` | Caddy configuration state |

## Notes

- Caddy automatically redirects HTTP to HTTPS when using domain names.
- For internal-only services, consider binding to a specific LAN interface instead of `0.0.0.0`.
- The Caddyfile is mounted read-only — edit it on the host, then reload.
