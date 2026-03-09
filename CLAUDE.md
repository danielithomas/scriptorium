# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Scriptorium is a collection of system optimization and administration scripts, organized by platform. Currently contains PowerShell scripts for Windows VM optimization, with a `linux/` directory planned for future expansion.

## Repository Structure

```
scriptorium/
├── windows/    # PowerShell scripts for Windows optimization
├── linux/      # Bash scripts for Linux (Ubuntu)
├── docker/     # Docker Compose stacks and services
│   └── image-gen/  # Multi-model image generation API (OpenVINO)
```

## Script Conventions

### PowerShell Scripts (windows/)

- Scripts use `#Requires -RunAsAdministrator` for elevated operations
- Use `Set-StrictMode -Version Latest` at the top of scripts
- Helper functions follow a consistent pattern: `Write-Section`, `Write-OK`, `Write-SKIP`, `Write-FAIL` for structured logging with color-coded console output and file logging
- Registry changes use the `Set-RegValue` wrapper (auto-creates parent keys, handles errors)
- Service changes use the `Disable-Service` wrapper (stops then disables, handles missing services)
- All destructive operations are wrapped in try-catch with status reporting
- Scripts create system restore points before making changes
- Output goes to both console (color-coded) and log files on Desktop

### Bash Scripts (linux/)

- Scripts use `#!/usr/bin/env bash` with `set -euo pipefail`
- Helper functions follow the same pattern: `write_section`, `write_ok`, `write_skip`, `write_fail`
- Scripts must be **re-runnable** (idempotent) — check before writing, skip already-applied changes, and only run install/update steps when needed
- Use `$REAL_USER` / `$REAL_HOME` (from `$SUDO_USER`) for user-owned files when running as root
- Use `id -u` instead of `$EUID` for POSIX compatibility

### Adding New Scripts

- Place platform-specific scripts in their respective directory (`windows/`, `linux/`)
- Follow the section-based structure with numbered sections and `write_section` headers
- Use the established helper functions for consistent logging and error handling
- Wrap operations in try-catch (PowerShell) or conditional checks (Bash) to report success/skip/fail rather than crashing
- Linux scripts must be re-runnable — never exit early in a way that skips configuration sections

### Docker Stacks (docker/)

- Each stack gets its own directory under `docker/`
- Every stack must include: `compose.yaml`, `Dockerfile` (if custom), `README.md`
- Use environment variables with defaults for host-specific paths (e.g., `${MODELS_PATH:-/data/models}`)
- **No PII** — this is a public repo. No hostnames, IPs, usernames, or identifying information
- Group IDs, device paths, and host-specific config should be commented examples, not hardcoded values
- README must include: quick start, model/data setup, API docs, environment variables

## Git Workflow

- `main` is the default branch
- `dev` branch for development — all new work goes here via PR
