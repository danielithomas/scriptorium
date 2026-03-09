# Image Generation Server (NVIDIA CUDA)

Multi-model image generation API using PyTorch + CUDA, with Real-ESRGAN upscaling and inpainting/outpainting. Optimised for NVIDIA GPUs.

## Features

- **Multiple models**: SD 1.5, DreamShaper 8, SDXL Turbo, SDXL 1.0, FLUX.1 Schnell
- **Auto-download**: Models download from HuggingFace on first use, then cache locally
- **FP16 inference**: Half precision by default for faster generation and lower VRAM
- **xformers**: Memory-efficient attention when available
- **Style presets**: photorealistic, anime, landscape, scifi, cute-dog
- **Aspect ratios**: square, wide (16:9), ultrawide (21:9), portrait
- **Upscaling**: Real-ESRGAN 4x on GPU
- **Outpainting**: Extend images using inpainting pipeline

## Requirements

- NVIDIA GPU with CUDA support (RTX 3060+ recommended, RTX 5080 ideal)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Docker with GPU support

### Install NVIDIA Container Toolkit (Windows + Docker Desktop)

Docker Desktop for Windows supports GPU passthrough natively with WSL2:
1. Ensure WSL2 is enabled with an Ubuntu distro
2. Install latest NVIDIA Game Ready drivers
3. Docker Desktop → Settings → Resources → WSL Integration → Enable for your distro
4. The `deploy.resources.reservations` in compose.yaml handles GPU access

## Quick Start

```bash
# 1. (Optional) Set models path for persistent storage
export MODELS_PATH=D:/models  # or /data/models on Linux

# 2. Build and run
docker compose up -d --build

# 3. First run will download the default model (~5GB for SD 1.5)
# Subsequent runs use local cache
```

## Model Registry

| Model | Key | Steps | Quality | Speed | VRAM |
|-------|-----|-------|---------|-------|------|
| SD 1.5 | `sd15` | 20 | Good | Fast | ~3GB |
| DreamShaper 8 | `dreamshaper` | 25 | Great | Fast | ~3GB |
| SDXL Turbo | `sdxl-turbo` | 4 | Good | Very Fast | ~5GB |
| SDXL 1.0 | `sdxl` | 30 | Excellent | Moderate | ~6GB |
| FLUX.1 Schnell | `flux-schnell` | 4 | Excellent | Fast | ~12GB |

Models download automatically on first use. To pre-download, set `MODELS_PATH` and run once.

### Local Model Storage

```
MODELS_PATH/
├── sd-v1-5/                    # SD 1.5 (auto-downloaded)
├── dreamshaper-8/              # DreamShaper 8 (auto-downloaded)
├── sdxl-turbo/                 # SDXL Turbo (auto-downloaded)
├── sdxl-base/                  # SDXL 1.0 (auto-downloaded)
├── flux-schnell/               # FLUX.1 Schnell (auto-downloaded)
├── sd15-inpainting/            # SD 1.5 Inpainting (auto-downloaded on first outpaint)
└── upscaler/
    └── RealESRGAN_x4plus.pth  # Download manually (~64MB)
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
    "model": "dreamshaper",
    "style_preset": "landscape",
    "aspect_ratio": "wide"
  }' | jq .seed,.elapsed_seconds,.model
```

### Health Check (shows GPU info)

```bash
curl -s http://localhost:8100/health | jq
# {
#   "status": "ok",
#   "device": "cuda",
#   "gpu": "NVIDIA GeForce RTX 5080",
#   "vram_total_gb": 16.0,
#   "vram_used_gb": 3.2,
#   "loaded_models": ["sd15"],
#   ...
# }
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_PATH` | `./models` | Host path to model directory |
| `DEFAULT_MODEL` | `sd15` | Default model key |
| `TORCH_DEVICE` | `cuda` | `cuda` or `cpu` |
| `HALF_PRECISION` | `true` | Use FP16 (faster, less VRAM) |

## Performance (estimated, RTX 5080)

| Model | Steps | Resolution | Est. Time |
|-------|-------|-----------|-----------|
| DreamShaper 8 | 25 | 512×512 | ~2-3s |
| SDXL Turbo | 4 | 1024×1024 | ~1-2s |
| SDXL 1.0 | 30 | 1024×1024 | ~8-12s |
| FLUX.1 Schnell | 4 | 1024×1024 | ~3-5s |

## Comparison: OpenVINO vs CUDA

| Feature | `image-gen` (OpenVINO) | `image-gen-cuda` (this) |
|---------|----------------------|------------------------|
| Target hardware | Intel CPU/iGPU/Arc | NVIDIA GPU |
| Model format | OpenVINO IR (pre-converted) | Native PyTorch/Safetensors |
| Auto-download | No (manual conversion) | Yes (HuggingFace) |
| FLUX support | No | Yes |
| Speed (SD 1.5) | ~67s (CPU) | ~2-3s (RTX 5080) |
| VRAM needed | N/A (CPU) | 3-12GB depending on model |
