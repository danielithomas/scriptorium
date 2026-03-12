# ComfyUI — NVIDIA CUDA (FLUX.1-dev)

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) node-based image generation with NVIDIA CUDA acceleration. Configured for **FLUX.1-dev** (12B parameters, fp8 quantized) on RTX 5080 / 16GB VRAM.

## Architecture

```
┌──────────────┐      POST /prompt       ┌──────────────────────┐
│ Orchestrator │ ──────────────────────►  │ ComfyUI (server-name)│
│              │      HTTP :8188          │  RTX 5080 · CUDA     │
│              │                          │                      │
└──────┬───────┘                          └──────────┬───────────┘
       │                                             │
       │  Poll every 30-60s                          │ Saves to
       │  for new files                              │ output dir
       │                                             │
       ▼                                             ▼
┌──────────────────────────────────────────────────────┐
│              Shared output directory                  │
│           (SMB share or local mount)                  │
└──────────────────────────────────────────────────────┘
```

### Flow

1. **Request** — Orchestrator (or any client) POSTs a workflow JSON to ComfyUI's API (`http://server-name:8188/prompt`)
2. **Queue** — ComfyUI queues the job internally; handles concurrency natively
3. **Generate** — GPU processes the workflow; FLUX.1-dev fp8 runs in ~15–30s per image at 1024×1024
4. **Save** — Output lands in the shared output directory
5. **Poll & Deliver** — Orchestrator polls the output directory every 30–60s; picks up new images and delivers them

### Why This Design

- **Decoupled** — generation and delivery are independent; no curl timeout pressure
- **Multi-client** — anyone on the LAN can queue work (agents, n8n, browser UI)
- **Resilient** — if the orchestrator is down, images still generate and wait in the share
- **Observable** — ComfyUI web UI at `:8188` for visual workflow editing and monitoring

## Prerequisites

- NVIDIA GPU with CUDA support (RTX 5080 tested, sm_120 / Blackwell)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Docker with GPU support
- ~18GB disk for FLUX.1-dev model files
- Hugging Face account (FLUX.1-dev requires license acceptance)

### Blackwell / RTX 50-Series Note

RTX 5080/5090/5070 GPUs use CUDA compute capability **sm_120** (Blackwell). The Dockerfile uses PyTorch nightly with CUDA 12.8 to support these GPUs, as stable PyTorch does not yet include sm_120 kernels.

## Quick Start

```bash
# 1. Accept the FLUX.1-dev license
#    https://huggingface.co/black-forest-labs/FLUX.1-dev
#    Then: huggingface-cli login

# 2. Install the HuggingFace CLI
pip install huggingface-hub[cli]

# 3. Download FLUX.1-dev model files (~18GB total)
python download-models.py /path/to/models          # FLUX components only
python download-models.py /path/to/models --all     # + embeddings, LoRAs, upscaler
python download-models.py --list                     # see all available downloads

# 4. Configure environment
cp .env.example .env
# Edit .env:
#   MODELS_PATH=/path/to/models
#   OUTPUT_PATH=/path/to/shared/output

# 5. Build and run
docker compose up -d --build

# 6. Open the UI
#    http://localhost:8188
```

### Shared Model Directory

If you point `MODELS_PATH` at an existing model directory (e.g., one shared with `image-gen-cuda`), ComfyUI will automatically see all LoRAs, embeddings, upscaler weights, and any `.safetensors` checkpoints in the appropriate subdirectories. The `extra_model_paths.yaml` maps the standard structure:

```
models/
├── checkpoints/    ← SD/SDXL .safetensors files (ComfyUI native format)
├── unet/           ← FLUX UNet weights
├── clip/           ← Text encoders (CLIP-L, T5-XXL)
├── vae/            ← VAE / autoencoder
├── loras/          ← LoRA fine-tunes (shared with image-gen-cuda)
├── embeddings/     ← Textual inversions (shared with image-gen-cuda)
└── upscaler/       ← Real-ESRGAN weights (shared with image-gen-cuda)
```

**Note:** `image-gen-cuda` downloads models in HuggingFace diffusers format (directories like `sdxl-base/`). ComfyUI expects single `.safetensors` checkpoint files in `checkpoints/`. The diffusers-format models won't appear in ComfyUI, but all shared auxiliary files (LoRAs, embeddings, upscaler) work across both.

## Model Files

The `download-models.py` script fetches everything needed:

| File | Location | Size | Description |
|------|----------|------|-------------|
| `flux1-dev-fp8.safetensors` | `unet/` | ~12GB | FLUX.1-dev diffusion model (fp8 quantized) |
| `clip_l.safetensors` | `clip/` | ~250MB | CLIP-L text encoder |
| `t5xxl_fp8_e4m3fn.safetensors` | `clip/` | ~5GB | T5-XXL text encoder (fp8 quantized) |
| `ae.safetensors` | `vae/` | ~335MB | FLUX autoencoder / VAE |

**Total: ~18GB**

### Why fp8?

Full FLUX.1-dev in bf16 requires ~24GB VRAM (UNet alone). The fp8 quantized version fits comfortably in 16GB with room for the VAE and text encoders, with minimal quality loss. This is the standard approach for RTX 4090/5080 class cards.

## Workflows

Pre-built workflow files in `workflows/`:

| Workflow | Resolution | Steps | Use Case |
|----------|-----------|-------|----------|
| `flux1-dev-t2i.json` | 1024×1024 | 20 | Standard text-to-image |
| `flux1-dev-t2i-ultrawide.json` | 1344×576 | 25 | Ultrawide wallpaper (21:9) |
| `flux1-dev-api-template.json` | Configurable | 20 | API automation template |

### Loading a Workflow

**Via UI:** Open ComfyUI → Load → select the `.json` file.

**Via API:** Use the api-template format:

```bash
curl -s http://server-name:8188/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": {
      "1": {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": "flux1-dev-fp8.safetensors", "weight_dtype": "fp8_e4m3fn"}
      },
      "2": {
        "class_type": "DualCLIPLoader",
        "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp8_e4m3fn.safetensors", "type": "flux"}
      },
      "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
      "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "YOUR PROMPT HERE", "clip": ["2", 0]}},
      "5": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
      "6": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["7", 0], "latent_image": ["5", 0], "seed": 0, "control_after_generate": "randomize", "steps": 20, "cfg": 3.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
      "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
      "8": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
      "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "flux1dev"}}
    }
  }'
```

The API returns a `prompt_id` immediately. The image generates asynchronously and appears in the output directory.

## Orchestrator Pattern

A separate orchestrator agent submits jobs and polls for results, decoupling generation from delivery.

### Submitting a Job

```python
import requests, json

COMFYUI_URL = "http://server-name:8188"

# Load the API template
with open("workflows/flux1-dev-api-template.json") as f:
    workflow = json.load(f)

# Remove the _comment key
workflow.pop("_comment", None)

# Customise the prompt
workflow["4"]["inputs"]["text"] = "a cyberpunk raven in neon rain"
workflow["5"]["inputs"]["width"] = 1024
workflow["5"]["inputs"]["height"] = 1024
workflow["6"]["inputs"]["steps"] = 20
workflow["6"]["inputs"]["seed"] = 42  # or 0 for random
workflow["9"]["inputs"]["filename_prefix"] = "my-job"

# Submit
response = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow})
prompt_id = response.json()["prompt_id"]
print(f"Queued: {prompt_id}")
```

### Polling for Output

```python
import os, time

OUTPUT_DIR = "/mnt/shared-output/comfyui-output"  # SMB mount or local path
POLL_INTERVAL = 30  # seconds

seen = set(os.listdir(OUTPUT_DIR))

while True:
    time.sleep(POLL_INTERVAL)
    current = set(os.listdir(OUTPUT_DIR))
    new_files = current - seen
    if new_files:
        for f in sorted(new_files):
            filepath = os.path.join(OUTPUT_DIR, f)
            print(f"New image: {filepath}")
            # → deliver via messaging, copy to another share, etc.
        seen = current
```

### Alternative: WebSocket Status

ComfyUI also supports WebSocket connections for real-time status updates. Connect to `ws://server-name:8188/ws` with the `client_id` from the prompt response for progress callbacks. This is more efficient than polling but requires maintaining a persistent connection.

## FLUX.1-dev Recommended Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| Steps | 20–28 | 20 is good, 28 for max quality |
| CFG | 3.5 | FLUX uses guidance distillation; low CFG is correct |
| Sampler | `euler` | Best results with FLUX |
| Scheduler | `simple` | Standard for FLUX |
| Resolution | 1024×1024 | Native; also supports 1344×576 (21:9), 768×1344 (portrait) |

**Note:** FLUX does not use a negative prompt in the traditional sense. The empty CLIPTextEncode node for `negative` is a ComfyUI requirement — leave it blank.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COMFYUI_VERSION` | `master` | Git branch/tag to build from |
| `COMFYUI_PORT` | `8188` | Host port for web UI and API |
| `MODELS_PATH` | `./models` | Host path to model directory |
| `OUTPUT_PATH` | `./output` | Host path for generated images |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU selection |

## Volumes

| Volume | Path | Description |
|--------|------|-------------|
| `MODELS_PATH` | `/data/models` | Model files (bind mount) |
| `OUTPUT_PATH` | `/app/output` | Generated images (bind mount to shared directory) |
| `comfyui_custom_nodes` | `/app/custom_nodes` | Installed custom node packages |
| `comfyui_user` | `/app/user` | User workflows and settings |
| `comfyui_input` | `/app/input` | Input images for img2img workflows |

## Performance Estimates (RTX 5080, 16GB)

| Resolution | Steps | Expected Time | VRAM Usage |
|-----------|-------|--------------|------------|
| 1024×1024 | 20 | ~15–20s | ~13GB |
| 1024×1024 | 28 | ~20–30s | ~13GB |
| 1344×576 | 25 | ~15–20s | ~12GB |
| 768×1344 | 25 | ~15–20s | ~12GB |

First generation includes model loading (~10–15s extra).

## Comparison: XPU vs CUDA

| Feature | `comfyui` (Intel XPU) | `comfyui-cuda` (this) |
|---------|----------------------|----------------------|
| Target hardware | Intel Arc / iGPU | NVIDIA GPU |
| FLUX support | Limited | Full (fp8) |
| Image format | Pre-built Docker image | Custom Dockerfile |
| Speed (FLUX.1-dev) | Very slow (CPU fallback) | ~15–20s (RTX 5080) |
| VRAM required | N/A (shared memory) | ~13GB |

## Troubleshooting

### sm_120 / Blackwell Warnings

If you see `sm_120 not compatible` warnings, the PyTorch build doesn't include Blackwell kernels. Rebuild the image to pick up the latest nightly:

```bash
docker compose build --no-cache
```

### Out of Memory

If you hit OOM errors, the fp8 model + VAE + text encoders may exceed 16GB in edge cases:

1. Close other GPU-using applications
2. Check VRAM: `nvidia-smi`
3. Consider using `t5xxl_fp8_e4m3fn` (already default) instead of the full T5-XXL

### Model Not Found

Ensure model files are in the correct subdirectories matching `extra_model_paths.yaml`:

```
models/
├── unet/flux1-dev-fp8.safetensors
├── clip/clip_l.safetensors
├── clip/t5xxl_fp8_e4m3fn.safetensors
└── vae/ae.safetensors
```
