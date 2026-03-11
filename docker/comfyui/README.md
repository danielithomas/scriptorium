# ComfyUI — Intel XPU

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) node-based image generation UI with Intel XPU (OpenVINO) acceleration. Uses the [reliq-hq XPU image](https://github.com/reliq-hq/ComfyUI) for Intel Arc / Meteor Lake iGPU support.

## Prerequisites

- Intel Arc GPU or integrated GPU with OpenVINO support
- `/dev/dri` device nodes accessible
- Docker with compose v2

## Quick Start

```bash
cp .env.example .env

# Create model directories
mkdir -p models/{checkpoints,embeddings,upscaler,loras}

# Find your GPU group IDs and update .env
getent group render  # Usually 992 or 106
getent group video   # Usually 44

docker compose up -d
```

Access the UI at `http://localhost:8188`.

## Model Directory Structure

Place your models in the host path configured by `MODELS_PATH`:

```
models/
├── checkpoints/     # SD 1.5, SDXL, etc.
├── embeddings/      # Textual inversions (e.g., EasyNegative)
├── upscaler/        # Real-ESRGAN, SwinIR, etc.
└── loras/           # LoRA weights
```

The `extra_model_paths.yaml` file maps these to ComfyUI's expected locations inside the container.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COMFYUI_PORT` | `8188` | Host port for web UI |
| `MODELS_PATH` | `./models` | Host path to shared model directory |
| `RENDER_GID` | `992` | Render group ID on host (for GPU access) |
| `VIDEO_GID` | `44` | Video group ID on host (for GPU access) |

## Volumes

| Volume | Description |
|--------|-------------|
| `comfyui_custom_nodes` | Installed custom node packages |
| `comfyui_output` | Generated images |
| `comfyui_user` | User workflows and settings |

## GPU Group IDs

The container needs access to GPU devices via Linux group membership. The correct GIDs vary by distribution:

```bash
# Find the right values for your system
getent group render
getent group video
stat -c '%g' /dev/dri/renderD128
```

Update `RENDER_GID` and `VIDEO_GID` in `.env` to match.

## Notes

- The `PYTORCH_ENABLE_XPU_FALLBACK=1` environment variable enables CPU fallback for unsupported XPU operations.
- Models are mounted read-only (`:ro`) — manage model files on the host.
- For NVIDIA GPU support, use the standard ComfyUI Docker image instead of the XPU variant.
- Custom nodes persist across container recreations via the named volume.
