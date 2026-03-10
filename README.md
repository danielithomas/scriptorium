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

Containerised service stacks for AI/ML workloads.

| Stack | Target Hardware | Description | Docs |
|-------|----------------|-------------|------|
| [`image-gen`](docker/image-gen/) | Intel CPU / iGPU / Arc | Multi-model image generation API using OpenVINO. Supports SD 1.5, DreamShaper LCM, and SDXL Turbo with Real-ESRGAN upscaling and outpainting. | [README](docker/image-gen/README.md) |
| [`image-gen-cuda`](docker/image-gen-cuda/) | NVIDIA GPU (CUDA) | Multi-model image generation API using PyTorch + CUDA. Supports SD 1.5, DreamShaper 8, SDXL Turbo, SDXL 1.0, and FLUX.1 Schnell with auto-download from HuggingFace. | [README](docker/image-gen-cuda/README.md) |

Both stacks expose the **same REST API** on port `8100`, making them drop-in replacements for each other. Choose based on your hardware:

- **Intel systems** → `image-gen` (OpenVINO, pre-converted models, works on CPU or Intel GPU)
- **NVIDIA systems** → `image-gen-cuda` (PyTorch, auto-downloads models, requires NVIDIA Container Toolkit)

#### Key files

| Stack | Files |
|-------|-------|
| `image-gen` | [`compose.yaml`](docker/image-gen/compose.yaml) · [`Dockerfile`](docker/image-gen/Dockerfile) · [`server.py`](docker/image-gen/server.py) |
| `image-gen-cuda` | [`compose.yaml`](docker/image-gen-cuda/compose.yaml) · [`Dockerfile`](docker/image-gen-cuda/Dockerfile) · [`server.py`](docker/image-gen-cuda/server.py) · [`download-models.py`](docker/image-gen-cuda/download-models.py) |

#### Quick comparison

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
