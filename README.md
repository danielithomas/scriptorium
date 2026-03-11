# Scriptorium

A collection of system administration scripts, optimisation tools, and Docker stacks — organised by platform.

## Scripts

### Windows

| Script | Description |
|--------|-------------|
| [`Optimise-Win11-OfficeVM.ps1`](windows/Optimise-Win11-OfficeVM.ps1) | Aggressively optimises a Windows 11 Pro VM for single-purpose Microsoft Office / Visio use. Removes bloat, disables telemetry, tunes services and UI. |

> **Warning:** The Windows optimisation script makes **irreversible changes** — it disables core services, removes built-in apps, and modifies system policies. It is designed exclusively for disposable, single-purpose VMs. **Do not run it on a daily-driver machine or any system you are not prepared to reinstall.** If you do not fully understand what it does, do not run it.

### Linux

| Script | Description |
|--------|-------------|
| [`Install-KDE-Plasma.sh`](linux/Install-KDE-Plasma.sh) | Installs KDE Plasma alongside GNOME on Ubuntu 25.10. Keeps GDM as the display manager, configures cross-toolkit theming, per-DE app associations, and autostart conflict prevention. |

### Docker

Containerised service stacks for self-hosted AI, monitoring, and infrastructure.

#### AI / ML

| Stack | Target Hardware | Description | Docs |
|-------|----------------|-------------|------|
| [`image-gen`](docker/image-gen/) | Intel CPU / iGPU / Arc | Multi-model image generation API (OpenVINO). SD 1.5, DreamShaper LCM, SDXL Turbo, Real-ESRGAN upscaling. | [README](docker/image-gen/README.md) |
| [`image-gen-cuda`](docker/image-gen-cuda/) | NVIDIA GPU (CUDA) | Multi-model image generation API (PyTorch + CUDA). SD 1.5, DreamShaper 8, SDXL Turbo, SDXL 1.0, FLUX.1 Schnell. | [README](docker/image-gen-cuda/README.md) |
| [`ollama`](docker/ollama/) | CPU / GPU | Standard Ollama LLM inference server. | [README](docker/ollama/README.md) |
| [`ollama-ipex`](docker/ollama-ipex/) | Intel Arc / iGPU | Ollama with IPEX-LLM for Intel GPU acceleration. | [README](docker/ollama-ipex/README.md) |
| [`comfyui`](docker/comfyui/) | Intel XPU | ComfyUI node-based image generation with Intel XPU support. | [README](docker/comfyui/README.md) |
| [`chatterbox-tts`](docker/chatterbox-tts/) | CPU / NVIDIA GPU | Chatterbox voice cloning TTS server (CPU and CUDA). | [README](docker/chatterbox-tts/README.md) |
| [`kokoro-tts`](docker/kokoro-tts/) | CPU | Lightweight OpenAI-compatible TTS server. | [README](docker/kokoro-tts/README.md) |
| [`opencode`](docker/opencode/) | CPU | AI coding assistant container connected to Ollama. | [README](docker/opencode/README.md) |

#### Infrastructure & Monitoring

| Stack | Description | Docs |
|-------|-------------|------|
| [`caddy`](docker/caddy/) | Caddy reverse proxy for LAN services. | [README](docker/caddy/README.md) |
| [`dockge`](docker/dockge/) | Docker Compose stack manager with web UI. | [README](docker/dockge/README.md) |
| [`n8n`](docker/n8n/) | Workflow automation platform. | [README](docker/n8n/README.md) |
| [`monitoring`](docker/monitoring/) | Uptime Kuma + Glances (service & system monitoring). | [README](docker/monitoring/README.md) |

#### Image Generation — Quick Comparison

`image-gen` and `image-gen-cuda` expose the **same REST API** on port `8100`, making them drop-in replacements. Choose based on hardware:

| | `image-gen` (OpenVINO) | `image-gen-cuda` (CUDA) |
|---|---|---|
| SDXL Turbo (1024×1024) | ~47s (CPU) | ~1–2s (RTX 5080) |
| Model format | OpenVINO IR (manual conversion) | Native PyTorch (auto-download) |
| FLUX support | No | Yes |
| VRAM required | N/A (CPU mode) | 3–12 GB depending on model |

## Usage

All shell scripts use a `#!/usr/bin/env bash` shebang. Run them with `./` (not `sh`) so the correct interpreter is used:

```bash
# Linux
chmod +x linux/Install-KDE-Plasma.sh
sudo ./linux/Install-KDE-Plasma.sh
```

```powershell
# Windows (run in an elevated PowerShell)
.\windows\Optimise-Win11-OfficeVM.ps1
```

```bash
# Docker (example: CUDA image generation)
cd docker/image-gen-cuda
python download-models.py /path/to/models
MODELS_PATH=/path/to/models docker compose up -d --build
```

> **Note:** Do not run `.sh` scripts with `sh script.sh` — they require Bash and will fail under other shells.
