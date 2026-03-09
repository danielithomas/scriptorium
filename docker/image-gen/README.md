# Image Generation Server

Multi-model image generation API using OpenVINO, with Real-ESRGAN upscaling and SD 1.5 inpainting/outpainting.

## Features

- **Multiple models**: SD 1.5, LCM DreamShaper v7 (fast), SDXL Turbo (high quality)
- **Style presets**: photorealistic, anime, landscape, scifi, cute-dog
- **Aspect ratios**: square, wide (16:9), ultrawide (21:9), portrait
- **Upscaling**: Real-ESRGAN 4x with optional target dimensions
- **Outpainting**: Extend images beyond borders using inpainting or img2img fallback
- **OpenVINO optimised**: Runs on CPU or Intel GPU (iGPU/Arc)

## Quick Start

```bash
# 1. Download models (see Model Setup below)
# 2. Set your models path
export MODELS_PATH=/path/to/your/models

# 3. Build and run
docker compose up -d --build
```

## Model Setup

Models go in `$MODELS_PATH` with this structure:

```
models/
├── sd-v1-5-openvino/          # SD 1.5 base (required)
├── lcm-dreamshaper-v7-int8-ov/ # DreamShaper LCM INT8 (optional, fast)
├── sdxl-turbo-openvino-8bit/   # SDXL Turbo INT8 (optional, quality)
├── sd15-inpainting-ov/         # SD 1.5 Inpainting (optional, for outpainting)
├── upscaler/
│   └── RealESRGAN_x4plus.pth  # Real-ESRGAN 4x (optional)
└── embeddings/                 # Textual inversion embeddings (optional)
```

### Converting models to OpenVINO format

```python
# SD 1.5
from optimum.intel import OVStableDiffusionPipeline
pipe = OVStableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", export=True)
pipe.save_pretrained("./sd-v1-5-openvino")

# DreamShaper LCM INT8
from optimum.intel import OVLatentConsistencyModelPipeline
pipe = OVLatentConsistencyModelPipeline.from_pretrained("deinferno/LCM_Dreamshaper_v7-openvino-int8")
pipe.save_pretrained("./lcm-dreamshaper-v7-int8-ov")

# SDXL Turbo (INT8 quantised)
from optimum.intel import OVStableDiffusionXLPipeline
pipe = OVStableDiffusionXLPipeline.from_pretrained("stabilityai/sdxl-turbo", export=True)
pipe.save_pretrained("./sdxl-turbo-openvino-8bit")
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate image, returns JSON with base64 |
| `/generate/image` | POST | Generate image, returns raw PNG |
| `/outpaint` | POST | Extend image beyond borders |
| `/upscale` | POST | Upscale image with Real-ESRGAN (multipart upload) |
| `/models` | GET | List available models and status |
| `/styles` | GET | List style presets |
| `/health` | GET | Service health check |

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

### Generate (raw image)

```bash
curl -s http://localhost:8100/generate/image \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cute puppy", "model": "dreamshaper"}' \
  -o output.png
```

### Upscale

```bash
curl -s http://localhost:8100/upscale \
  -F "image=@input.png" \
  -F "target_width=2048" \
  -o upscaled.png
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_PATH` | `/data/models` | Host path to model directory |
| `OV_DEVICE` | `CPU` | OpenVINO device: `CPU`, `GPU`, or `AUTO` |
| `DEFAULT_MODEL` | `sd15` | Default model: `sd15`, `dreamshaper`, `sdxl-turbo` |

## GPU Support

For Intel iGPU/Arc acceleration, ensure:
1. The `/dev/dri` device is passed through (already in compose)
2. Set `OV_DEVICE=GPU` or `OV_DEVICE=AUTO`
3. Uncomment `group_add` in compose.yaml and set your host's render/video group IDs:
   ```bash
   getent group render video | cut -d: -f3
   ```

For NVIDIA GPUs, this stack uses OpenVINO (Intel-optimised). For NVIDIA, consider using the standard PyTorch/CUDA diffusers pipeline instead.

## Performance (approximate, CPU)

| Model | Steps | Resolution | Time |
|-------|-------|-----------|------|
| DreamShaper LCM | 4 | 512×512 | ~10s |
| SD 1.5 | 20 | 512×512 | ~67s |
| SDXL Turbo | 4 | 1024×1024 | ~47s |

GPU acceleration (Intel Arc) can significantly reduce these times.
