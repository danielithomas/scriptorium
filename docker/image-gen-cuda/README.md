# Image Generation Server (NVIDIA CUDA)

Multi-model image generation API using PyTorch + CUDA, with Real-ESRGAN upscaling and inpainting/outpainting. Optimised for NVIDIA GPUs.

## Features

- **Multiple models**: SD 1.5, DreamShaper 8, SDXL Turbo, SDXL 1.0, FLUX.1 Schnell
- **Auto-download**: Models download from HuggingFace on first use, then cache locally
- **FP16 inference**: Half precision by default for faster generation and lower VRAM
- **xformers**: Memory-efficient attention when available
- **Style presets**: photorealistic, anime, landscape, scifi, cute-dog
- **Aspect ratios**: square, wide (16:9), ultrawide (21:9), portrait
- **Upscaling**: Real-ESRGAN 4x on GPU (supports target resolution, e.g. 3440×1440)
- **Outpainting**: Extend images using inpainting pipeline
- **Asset downloader**: `download-models.py` fetches additional models, LoRAs and embeddings for use in other tools (e.g. [comfyui-cuda](../comfyui-cuda/))

## Requirements

- NVIDIA GPU with CUDA support (RTX 3060+ recommended)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Docker with GPU support

### Blackwell / RTX 50-Series Note

RTX 5080/5090/5070 GPUs use CUDA compute capability **sm_120** (Blackwell architecture). As of early 2026, **PyTorch stable does not include sm_120 kernels**. The Dockerfile uses PyTorch nightly with CUDA 12.8 to support these GPUs. If you see warnings about `sm_120 not compatible`, ensure you've rebuilt the image with the latest Dockerfile.

### Install NVIDIA Container Toolkit (Windows + Docker Desktop)

Docker Desktop for Windows supports GPU passthrough natively with WSL2:
1. Ensure WSL2 is enabled with an Ubuntu distro
2. Install latest NVIDIA Game Ready drivers
3. Docker Desktop → Settings → Resources → WSL Integration → Enable for your distro
4. The `deploy.resources.reservations` in compose.yaml handles GPU access

## Quick Start

```bash
# 1. Install Python dependencies for the download script
pip install diffusers transformers accelerate torch safetensors

# 2. Download models to your chosen directory
python download-models.py D:\SD\models          # required models only (~5GB)
python download-models.py D:\SD\models --all    # all models + LoRAs (~80GB)

# 3. Create .env file with your models path
cp .env.example .env
# Edit .env → MODELS_PATH=D:\SD\models

# 4. Build and run
docker compose up -d --build
```

### Download Script

```bash
# List all available models, LoRAs, and embeddings
python download-models.py --list

# Download specific models
python download-models.py D:\SD\models --models sd-v1-5 dreamshaper-8 sdxl-turbo

# Download all models plus LoRAs
python download-models.py D:\SD\models --all

# Download LoRAs only (alongside default required models)
python download-models.py D:\SD\models --loras

# Re-download / update a model
python download-models.py D:\SD\models --models sd-v1-5 --force
```

The script **skips models that already exist** locally — safe to run repeatedly to add new models over time. Models are saved in standard HuggingFace diffusers format.

### Configuration via `.env`

Create a `.env` file in this directory (see `.env.example`):

```env
MODELS_PATH=/path/to/your/models
```

Docker Compose reads `.env` automatically. The `.env` file is gitignored (machine-specific paths).

## Model Registry

Two registries exist and they are **not** the same list:

- **`MODELS` in `server.py`** — what the API can actually serve. The `model` field of a request must be one of these keys; anything else returns a 400.
- **`MODELS` / `LORAS` / `EMBEDDINGS` in `download-models.py`** — what can be fetched to disk. This is a superset, and doubles as the asset downloader for [comfyui-cuda](../comfyui-cuda/), which shares the same `loras/`, `embeddings/` and `upscaler/` directories.

### Servable Models (API keys)

| Model | API key | Steps | Quality | Speed (RTX 5080) | VRAM |
|-------|---------|-------|---------|-------------------|------|
| SD 1.5 | `sd15` | 20 | Good | ~2s | ~3GB |
| DreamShaper 8 | `dreamshaper` | 25 | Great | ~3s | ~3GB |
| SDXL Turbo | `sdxl-turbo` | 4 | Good | ~5s | ~5GB |
| SDXL 1.0 Base | `sdxl` | 30 | Excellent | ~12s | ~6GB |
| FLUX.1 Schnell | `flux-schnell` | 4 | Excellent | ~5s | ~12GB |

SD 1.5 Inpainting is loaded separately and automatically by `/outpaint` — it is not a `model` key.

`GET /models` returns this list with per-model `local` (cached on disk) and `loaded` (in VRAM) flags.
A model that is not cached locally is downloaded from HuggingFace on first use.

### Additional Downloads (not servable by this API)

`download-models.py` can also fetch the following. They are **not** in `server.py`'s registry — downloading one does not make it available to `/generate`; see [Adding New Models](#adding-new-models).

| Download key | Notes |
|--------------|-------|
| `realistic-vision-6` | Realistic Vision v6 (SD 1.5 based, photo) |
| `sdxl-refiner` | SDXL Refiner — a post-process stage, not a standalone model |
| `juggernaut-xl` | Juggernaut XL v9 (SDXL, photo) |
| `dreamshaper-xl` | DreamShaper XL |
| `realvis-xl-4` | RealVisXL v4 (SDXL, ultra-photo) |
| `flux-dev` | FLUX.1 Dev — ~24GB VRAM, exceeds a 16GB card |

### LoRAs (Lightweight Fine-Tunes)

Downloaded to `MODELS_PATH/loras/` for use in ComfyUI or other tooling. **The `image-gen-cuda` API does not apply LoRAs** — there is no LoRA parameter on `/generate`.

| LoRA | Base Model | Source | Description |
|------|-----------|--------|-------------|
| SDXL Lightning 4-step | SDXL | HuggingFace | ByteDance fast generation LoRA |
| Detail Tweaker XL | SDXL | CivitAI (manual) | Micro-detail and sharpness |
| Film Grain XL | SDXL | CivitAI (manual) | Cinematic film grain |

### Embeddings (Textual Inversions)

Tiny files (~25KB) downloaded to `MODELS_PATH/embeddings/`. Like LoRAs, they are **not** loaded by this API — the negative-prompt tokens below only take effect in a tool that loads the embedding files (e.g. ComfyUI).

| Embedding | Base Model | Usage (in negative prompt) |
|-----------|-----------|---------------------------|
| EasyNegative | SD 1.5 | `EasyNegative` |
| bad-hands-5 | SD 1.5 | `bad-hands-5` |
| NegativeXL | SDXL | `negativeXL_D` |

### Local Model Storage

```
MODELS_PATH/
├── sd-v1-5/                    # SD 1.5
├── dreamshaper-8/              # DreamShaper 8
├── realistic-vision-6/         # Realistic Vision v6
├── sdxl-turbo/                 # SDXL Turbo
├── sdxl-base/                  # SDXL 1.0 Base
├── sdxl-refiner/               # SDXL Refiner
├── juggernaut-xl/              # Juggernaut XL v9
├── dreamshaper-xl/             # DreamShaper XL
├── realvis-xl-4/               # RealVisXL v4
├── flux-schnell/               # FLUX.1 Schnell
├── flux-dev/                   # FLUX.1 Dev
├── sd15-inpainting/            # SD 1.5 Inpainting
├── loras/
│   ├── sdxl-lightning-4step.safetensors
│   ├── detail-tweaker-xl.safetensors   # manual download
│   └── film-grain-xl.safetensors       # manual download
├── embeddings/
│   ├── EasyNegative.safetensors
│   ├── bad-hands-5.pt
│   └── negativeXL_D.safetensors
└── upscaler/
    └── RealESRGAN_x4plus.pth
```

## API Endpoints

Same API as the OpenVINO variant — drop-in replacement.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate image, returns JSON with base64 |
| `/generate/image` | POST | Generate image, returns raw PNG |
| `/outpaint` | POST | Extend image beyond borders |
| `/upscale` | POST | Upscale with Real-ESRGAN (multipart upload) |
| `/models` | GET | List models and status |
| `/styles` | GET | List style presets |
| `/health` | GET | Health check with GPU/VRAM info |

### Generate

```bash
curl -s http://localhost:8100/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a serene mountain landscape at sunset",
    "negative_prompt": "low quality, blurry, watermark, text",
    "model": "sdxl",
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "guidance_scale": 7.5
  }' | jq .seed,.elapsed_seconds,.model
```

### Generate Ultrawide Wallpaper (3440×1440)

```bash
curl -s http://localhost:8100/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "sweeping mountain valley, golden hour, photorealistic, 8k",
    "negative_prompt": "low quality, blurry, pixelated, text, watermark",
    "model": "sdxl",
    "width": 1344,
    "height": 576,
    "steps": 30,
    "guidance_scale": 7.5,
    "upscale": true,
    "upscale_target_width": 3440,
    "upscale_target_height": 1440
  }'
```

Generate at native 21:9 (1344×576), then upscale via Real-ESRGAN to 3440×1440.
Expect ~16 minutes total (generation + 4× upscale) — worth it for wallpaper-quality output.

### Health Check

```bash
curl -s http://localhost:8100/health | jq
# {
#   "status": "ok",
#   "device": "cuda",
#   "gpu": "NVIDIA GeForce RTX 5080",
#   "vram_total_gb": 15.9,
#   "vram_used_gb": 3.2,
#   "loaded_models": ["sd15"],
#   ...
# }
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_PATH` | `./models` | Host path to model directory |
| `DEFAULT_MODEL` | `sd15` | Default model key to load on startup |
| `TORCH_DEVICE` | `cuda` | `cuda` or `cpu` |
| `HALF_PRECISION` | `true` | Use FP16 (faster, less VRAM) |

## Recommended Negative Prompt

For best results, include a universal negative prompt with all generations:

```
low quality, worst quality, lowres, blurry, pixelated, jpeg artifacts, text,
watermark, logo, signature, extra fingers, extra limbs, missing limbs, deformed
hands, bad hands, mutated, distorted, disfigured, bad anatomy, bad proportions,
bad perspective, cartoon, anime, 3d render, cgi
```

Use embeddings for even better results — add `EasyNegative` (SD 1.5) or `negativeXL_D` (SDXL) to the negative prompt alongside the text.

## Performance (Measured, RTX 5080 16GB)

| Model | Steps | Resolution | Time | Notes |
|-------|-------|-----------|------|-------|
| SD 1.5 | 20 | 512×512 | **1.9s** | Fastest |
| SDXL Turbo | 4 | 1024×1024 | **5.2s** | Best speed/quality |
| SDXL 1.0 | 30 | 1024×1024 | ~12s | High quality |
| SDXL 1.0 | 30 | 1344×576 + upscale 3440×1440 | ~16min | Wallpaper quality |

First generation with a new model includes a model load (~5–15s extra).

## Adding New Models

Making a model servable takes **two** registry edits — the download script and the server do not share a registry:

1. Add the model to the `MODELS` dict in `download-models.py` (HuggingFace id + local path)
2. Download it (existing models are skipped): `python download-models.py <models-dir> --models <key>`
3. Add a matching entry to the `MODELS` dict in `server.py` with `model_id`, `local_path`, `type`
   (`sd15`, `sdxl` or `flux` — this selects the diffusers pipeline class), `description`,
   `default_steps` and `default_guidance`
4. Restart the container — `server.py` is bind-mounted, so no rebuild is needed

Skipping step 3 means the weights sit on disk but `/generate` rejects the key.

For LoRAs and embeddings, add to the `LORAS` / `EMBEDDINGS` dicts in `download-models.py`. They land in
`MODELS_PATH/loras/` and `MODELS_PATH/embeddings/` for ComfyUI to consume; this API will not apply them.

## Comparison: OpenVINO vs CUDA

| Feature | `image-gen` (OpenVINO) | `image-gen-cuda` (this) |
|---------|----------------------|------------------------|
| Target hardware | Intel CPU/iGPU/Arc | NVIDIA GPU |
| Model format | OpenVINO IR (pre-converted) | Native PyTorch/Safetensors |
| Auto-download | No (manual conversion) | Yes (HuggingFace) |
| FLUX support | No | Yes (Schnell) |
| LoRA support | No | Not at inference — downloader only |
| Speed (SD 1.5) | ~67s (CPU) | **~2s** (RTX 5080) |
| VRAM needed | N/A (CPU) | 3–12GB depending on model |

Both expose the same endpoints on port 8100 and are drop-in replacements for one another.
