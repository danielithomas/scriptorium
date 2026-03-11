# n8n

[n8n](https://n8n.io) — workflow automation platform. Build complex automations with a visual node-based editor connecting APIs, databases, and services.

## Quick Start

```bash
cp .env.example .env
# Edit TZ to your timezone

mkdir -p data
docker compose up -d
```

Access the UI at `http://localhost:5678`. First-time setup will prompt for an admin account.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `N8N_PORT` | `5678` | Host port for web UI |
| `N8N_DATA` | `./data` | Host path for workflow data and credentials |
| `TZ` | `UTC` | Timezone for scheduled workflows |

## Data Persistence

All workflows, credentials, and settings are stored in the `data/` directory (mapped to `/home/node/.n8n` inside the container).

**Back up this directory regularly** — it contains your workflow definitions and encrypted credentials.

## Configuration Notes

- `N8N_SECURE_COOKIE=false` — Required when accessing over HTTP (no TLS). Set to `true` if behind an HTTPS reverse proxy.
- `N8N_PROTOCOL=http` — Change to `https` if terminating TLS at n8n directly.
- For production use, consider adding a database backend (PostgreSQL) instead of the default SQLite.

## Useful n8n Features

- **Webhook triggers** — Receive HTTP callbacks to start workflows
- **Cron triggers** — Schedule workflows on a timer
- **200+ integrations** — Google Sheets, Slack, GitHub, databases, and more
- **Code nodes** — Write JavaScript or Python for custom logic

## Security Notes

- n8n stores credentials encrypted, but the encryption key is in the data directory.
- Do not expose n8n to the internet without authentication and HTTPS.
- Consider placing behind a reverse proxy (see [caddy](../caddy/) stack).
