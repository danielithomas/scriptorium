# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Scriptorium is a collection of system administration scripts and self-hosted Docker stacks, organised by platform: `windows/` (PowerShell), `linux/` (Bash), `docker/` (Compose stacks, a few with custom Python services).

There is no build system, package manager, test suite, or linter at the repo root. "Running" something means executing a script directly or bringing up a Compose stack. Each `docker/<stack>/README.md` is the authoritative reference for that stack — read it before changing the stack.

## Commands

```bash
# Validate a Compose file (variable interpolation + schema) without starting anything
docker compose -f docker/<stack>/compose.yaml config

# Bring a stack up (run from inside the stack directory so its .env is picked up)
cd docker/<stack> && docker compose up -d --build

# Stacks with CPU/GPU variants select the file explicitly
cd docker/slideshow-gen && docker compose -f compose-gpu.yaml up -d --build

# Syntax-check a Bash script without running it
bash -n linux/Install-KDE-Plasma.sh

# Model downloads are separate, idempotent, and run on the HOST (not in the container)
python docker/image-gen-cuda/download-models.py /path/to/models     # --sd15 --sdxl --flux ...
python docker/comfyui-cuda/download-models.py /path/to/models       # --fill --flux2 --checkpoints --extras
```

Editing `server.py` in `image-gen`, `image-gen-cuda`, or `slideshow-gen` does **not** require a rebuild — it is bind-mounted read-only into the container, so `docker compose restart` suffices. Dependency or Dockerfile changes need `--build`.

## Architecture

### `image-gen` and `image-gen-cuda` — parallel implementations of one API

Two independent FastAPI services exposing the **same REST API on port 8100**, intended as drop-in replacements for each other. `image-gen` runs OpenVINO (Intel CPU/iGPU/Arc); `image-gen-cuda` runs PyTorch/CUDA (NVIDIA).

Both `server.py` files share a structure: a module-level `MODELS` registry (key → path/model_id, type, `default_steps`, `default_guidance`), `STYLE_PRESETS` (prompt prefix + negative prompt), `ASPECT_RATIOS`, lazy `load_pipeline()` caching into a global `pipelines` dict, and a **single-worker `ThreadPoolExecutor`** so accelerator work is serialised behind the async endpoints. Endpoints: `POST /generate`, `/generate/image`, `/outpaint`, `/upscale`; `GET /models`, `/styles`, `/health`.

**When changing API surface, request/response models, style presets, or aspect ratios, mirror the change in both files** — drift breaks the drop-in property the README advertises. Backend-specific code (`OV_DEVICE` vs `TORCH_DEVICE`/`HALF_PRECISION`, model formats, the OpenVINO img2img outpaint fallback) legitimately differs.

### `slideshow-gen` — six-stage pipeline with two front-ends

`scripts/stage_*.py` are the pipeline. They are driven by either front-end and communicate through a shared working directory rather than function calls:

1. **Narration Synthesis** — runs **first**, because it writes `durations.json`, which drives every subsequent slide timing. Routes English → Chatterbox, Hindi/Punjabi/Gujarati → IndicF5 (imported in-process; it does not call the `chatterbox-tts` stack over HTTP).
2. Image Preparation (ImageMagick) → 3. Music Generation (MusicGen) → 4. Audio Mixing → 5. Video Segments + Overlays → 6. Final Assembly (FFmpeg).

Front-ends:
- `server.py` — persistent FastAPI job service. Multipart `POST /generate` writes a job into `/data/jobs/{job_id}/`; a **single daemon worker thread** pulls from a `Queue` and runs the stages sequentially. `status.json` on disk is the source of truth for progress, so jobs interrupted by a restart are re-queued at startup by `_restore_jobs()`. Old jobs are pruned past `MAX_STORED_JOBS`.
- `entrypoint.sh` — legacy one-shot CLI. Parses flags, exports them as env vars, and invokes the same six stage scripts in order.

A new stage or a changed stage contract must be updated in **both** `server.py` (the `STAGES` list and `_run_pipeline`) and `entrypoint.sh`.

### `comfyui-cuda` — no custom service

Runs upstream ComfyUI; the repo's contribution is `workflows/*.json`, `download-models.py`, and `extra_model_paths.yaml`. Workflows named `*-api-template.json` are the machine-callable form (POST to `/prompt`); the rest are UI workflows. `MODELS_PATH` is deliberately shareable with `image-gen-cuda`: `loras/`, `embeddings/`, and `upscaler/` are common to both, but the diffusers-format directories `image-gen-cuda` downloads are invisible to ComfyUI, which needs single `.safetensors` files in `checkpoints/`.

The intended deployment pattern is decoupled: a client POSTs a workflow, ComfyUI queues it and writes to a shared output directory, and the orchestrator polls that directory — no long-lived HTTP request holds the generation open.

### `download-models.py` scripts

Registry dicts at the top (`MODELS`, `CHECKPOINTS`, `LORAS`, `EMBEDDINGS`, `UPSCALER`, per-model `*_COMPONENTS`) plus argparse flags selecting which registries to fetch. They use the same `print_ok`/`print_skip`/`print_fail` reporting idiom as the shell and PowerShell scripts, and must stay idempotent — check for an existing model directory and skip rather than re-download. Adding a model means adding a registry entry, not new download code.

## Conventions

### Docker stacks (docker/)

- Each stack has its own directory containing `compose.yaml` (or `compose-cpu.yaml`/`compose-gpu.yaml`), a `Dockerfile` if custom, `.env.example`, and `README.md`
- Host-specific values go through `${VAR:-default}` interpolation, never hardcoded; required values use `${VAR:?message}`
- Commit `.env.example`; `.env` and `**/data/` are gitignored — runtime data (models, databases, job dirs) must never be committed
- Set `container_name` and `restart: unless-stopped`
- GPU access: NVIDIA via `deploy.resources.reservations.devices`, Intel via `devices: /dev/dri`. Host-specific group IDs (`RENDER_GID`, `VIDEO_GID`, `PUID`/`PGID`) belong in `.env.example` with the discovery command in a comment (`getent group render video`), and stay commented out in `compose.yaml`
- README must include: quick start, model/data setup, API docs, environment variables
- **No PII** — this is a public repo. No hostnames, IPs, usernames, tokens, or identifying paths. Use `server-name`, `localhost`, `/path/to/models`

Default host ports in use: 80/443 caddy · 3001 uptime-kuma · 5001 dockge · 5678 n8n · 6767 bazarr · 8004 chatterbox-tts · 8096 jellyfin · 8100 image-gen(-cuda) · 8110 iopaint-cuda · 8188 comfyui(-cuda) · 8189 slideshow-gen · 8880 kokoro-tts · 11434 ollama · 11435 ollama-ipex · 61208 glances.

### PowerShell scripts (windows/)

- `#Requires -RunAsAdministrator` for elevated operations; `Set-StrictMode -Version Latest` at the top
- Numbered sections via `Write-Section`; per-action results via `Write-OK` / `Write-SKIP` / `Write-FAIL`, which write colour-coded console output *and* append to a log file on the Desktop
- Registry changes use the `Set-RegValue` wrapper (auto-creates parent keys, handles errors); service changes use `Disable-Service` (stops then disables, tolerates missing services)
- Create a system restore point before destructive work, and wrap operations in try-catch so a failure reports `[FAIL]` rather than aborting the run

### Bash scripts (linux/)

- `#!/usr/bin/env bash` with `set -euo pipefail`. Scripts require Bash — never invoke with `sh`
- Same reporting idiom: `write_section`, `write_ok`, `write_skip`, `write_fail`
- Must be **re-runnable** (idempotent) — check before writing, skip already-applied changes, and never exit early in a way that skips later configuration sections
- Use `$REAL_USER` / `$REAL_HOME` (from `$SUDO_USER`) for user-owned files when running as root, and `id -u` instead of `$EUID`

## Git Workflow

- `main` is the default branch; all new work goes to `dev` and reaches `main` via PR
- Commit messages are Conventional Commits scoped to the stack or script: `feat(slideshow-gen): narration-driven slide timing`, `fix(ollama): ...`

### PII check — required before every `git add` and `git push`

This is a public repo. **Run this sweep before staging or pushing, every time**, and fix what it
finds before committing. Do not treat a clean `.gitignore` as sufficient — most of what leaks here
is prose in READMEs and defaults in compose files, not stray files.

```bash
# Identity, host paths, network addresses
git grep -nEi "danielithomas|Daniel Thomas|theenquiringmind|@(gmail|outlook|hotmail)" -- .
git grep -nE  "C:\\\\Users\\\\|/home/[a-z][a-z0-9_-]+|/Users/[a-z]" -- .
git grep -nE  "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b" -- .        # 0.0.0.0 / 127.0.0.1 are fine

# Credentials
git grep -nEi "(api[_-]?key|secret|password|token|credential)\s*[:=]\s*[\"']?\S{8,}" -- .
git grep -nE  "hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}" -- .

# Host-specific config that should be a generic example.
# Both patterns are deliberately narrow — the obvious broad versions drown in
# false positives (registry hives HKLM:\… and IANA examples in prose).
git grep -nE  "\b[C-Z]:[\\\\/]" -- . | grep -viE "HK(LM|CU|CR|U|CC):"   # drive paths
git grep -nE  "TZ.?=.?[\"']?(Africa|America|Asia|Australia|Europe|Pacific)/" -- .  # TZ default

# Files that should never be tracked (checks all history, not just HEAD)
git log --all --pretty=format: --name-only --diff-filter=A | sort -u \
  | grep -Ei "(^|/)\.env$|/data/|\.db$|\.sqlite|\.pem$|\.key$|id_rsa"
git ls-files -ci --exclude-standard                          # ignored but tracked anyway
```

Beyond the greps, watch for **aggregation** — individually harmless details that compose into a
personal profile. A timezone default, media-library folder names, example film titles and TTS
language choices together identify a person's location, household and interests far more precisely
than any one of them does. Judge the set, not the line.

Fixes are always the same shape: replace with a neutral placeholder (`/path/to/models`,
`Example Movie (2019)`, `TZ=UTC`, `server-name`) and keep the real value in `.env`, which is
gitignored. Real values belong in commented examples only when the comment explains how to derive
them for *your own* host (`getent group render`), never as the value itself.

Exceptions that are fine: the committer identity in git metadata, and the `Co-Authored-By` /
`Claude-Session` trailers.
