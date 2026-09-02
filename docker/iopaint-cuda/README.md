# iopaint-cuda

Docker Compose stack for [IOPaint](https://github.com/Sanster/IOPaint) — an open-source image inpainting and object-removal tool — with NVIDIA CUDA GPU acceleration.

For diffusion-based outpainting and inpainting driven by prompts, see [comfyui-cuda](../comfyui-cuda/) (FLUX.1-Fill-dev) or [image-gen-cuda](../image-gen-cuda/) (`/outpaint`). IOPaint is the better tool for erasing objects and cleaning up images without a prompt.

## Hardware

- **NVIDIA GPU** (CUDA ≥ 11.8 recommended)
- **Driver:** nvidia-container-toolkit configured

## Quick Start

### 1. Set up environment

```bash
cp .env.example .env
# Edit .env — set MODEL_DIR to your host model cache directory
```

`MODEL_DIR` must point to the host directory that holds your HuggingFace and Torch caches. The
container runs as **root**, so this directory must be writable by root on the Docker host.

### 2. Download models (optional — iopaint downloads models automatically at startup)

If you prefer to pre-cache models on the host:

```bash
export MODEL_DIR=/path/to/iopaint/models

python3 download-models.py              # LaMa only — the default model (~196MB)
python3 download-models.py --diffusers  # + diffusers inpainting models (~7GB)
python3 download-models.py --list       # show what's available
python3 download-models.py --force      # re-download even if present
```

The default run needs no dependencies beyond the standard library. `--diffusers` additionally
requires `pip install huggingface-hub`.

Only download the diffusers models if you intend to change `IOPAINT_MODEL` — with the default
`lama` they are never loaded.

### 3. Launch

```bash
MODEL_DIR=/path/to/iopaint/models docker compose up -d
```

Access at **http://localhost:8110** (or whatever `IOPAINT_PORT` is set to).

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | _(required*)_ | Host path the two caches derive from |
| `HF_CACHE_DIR` | `$MODEL_DIR/huggingface` | HuggingFace `HF_HOME` — the directory containing `hub/` |
| `TORCH_CACHE_DIR` | `$MODEL_DIR/torch` | Torch cache root — the directory containing `hub/checkpoints/` |
| `IOPAINT_PORT` | `8110` | Host port the stack listens on |
| `IOPAINT_MODEL` | `lama` | Model loaded at startup (lama, migan, zdiff, or a diffusers repo id) |
| `IOPAINT_DEVICE` | `cuda` | Inference device — set `cpu` to run without a GPU |
| `IOPAINT_EXTRA_ARGS` | _(empty)_ | Additional `iopaint start` CLI flags |

## Models

iopaint downloads models at startup if not already cached. The two bind mounts put those caches on
the host at the exact paths the libraries read from inside the container:

| Host path | Container path | Holds |
|-----------|----------------|-------|
| `MODEL_DIR/torch/hub/checkpoints/` | `/root/.cache/torch/hub/checkpoints/` | Single-file models fetched by URL (LaMa, MI-GAN, …) |
| `MODEL_DIR/huggingface/hub/` | `/root/.cache/huggingface/hub/` | Diffusers-format models, in HuggingFace cache layout |

The `hub/` segment in both paths is required — it is where `torch.hub` and `huggingface_hub` look by
default. A model placed one level up is invisible to the container and gets downloaded again at
startup.

`download-models.py` writes to exactly these paths, so anything it fetches is picked up on the next
`docker compose up`.

\* `MODEL_DIR` is only required when `HF_CACHE_DIR` and `TORCH_CACHE_DIR` are not both set.

### Sharing a cache with other stacks

[`slideshow-gen`](../slideshow-gen/) already sets `HF_HOME=/models` and `TORCH_HOME=/models/torch`
against its shared model root, so pointing iopaint at the same directories makes both stacks reuse
one cache instead of downloading the same weights twice:

```env
# .env — MODEL_DIR is unused in this mode and may be omitted
HF_CACHE_DIR=D:/SD/models
TORCH_CACHE_DIR=D:/SD/models/torch
```

That resolves to `D:/SD/models/hub` and `D:/SD/models/torch/hub/checkpoints` — the same layout
slideshow-gen produces. Export the same two variables when running `download-models.py` so the
host-side downloads land in the shared cache too.

Note this shares the *HuggingFace and torch caches only*. The structured model tree that
[`comfyui-cuda`](../comfyui-cuda/) and [`image-gen-cuda`](../image-gen-cuda/) use (`checkpoints/`,
`unet/`, `loras/`, …) is a different layout, so no weights are shared with those two — iopaint just
adds `hub/`, `xet/` and `torch/` alongside them.

### Windows hosts

The HuggingFace cache layout uses symlinks and long directory names. If `--diffusers` fails with
`WinError 3` or a symlink permission error, either keep `MODEL_DIR` short (e.g. `D:\iopaint`) or
enable Developer Mode so unprivileged symlink creation is allowed. The default LaMa download is a
plain file copy and is unaffected.

## Container Command

The image's entrypoint (`/opt/nvidia/nvidia_entrypoint.sh`) performs the CUDA environment setup, so
the compose file overrides only the **arguments**, not the entrypoint. If you need to change how
iopaint starts, adjust `IOPAINT_MODEL` / `IOPAINT_DEVICE` / `IOPAINT_EXTRA_ARGS` rather than
replacing `entrypoint:` — doing so skips the NVIDIA setup.

## NVIDIA Runtime

The compose file requests the GPU through `deploy.resources.reservations.devices` with `driver: nvidia`,
which is the same mechanism used by the other CUDA stacks in this repo and requires
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
on the host.

If Docker cannot find the NVIDIA runtime, set it as the default in `/etc/docker/daemon.json`:

```json
{
  "default-runtime": "nvidia"
}
```
