# ComfyUI — NVIDIA CUDA (Multi-Model)

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) node-based image generation with NVIDIA CUDA acceleration. Supports **FLUX.1-dev**, **FLUX.2-klein-9B**, **SDXL**, **FLUX.1-Kontext-dev**, **HiDream-I1**, **HiDream-O1-Image**, and **Qwen-Image** on RTX 5080 / 16GB+ VRAM.

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
3. **Generate** — GPU processes the workflow
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
- Hugging Face account (gated models require license acceptance)
- `pip install huggingface-hub[cli]`

### Blackwell / RTX 50-Series Note

RTX 5080/5090/5070 GPUs use CUDA compute capability **sm_120** (Blackwell). The Dockerfile uses PyTorch nightly with CUDA 12.8 to support these GPUs, as stable PyTorch does not yet include sm_120 kernels.

## Quick Start

```bash
# 1. Install the HuggingFace CLI and log in
pip install huggingface-hub[cli]
huggingface-cli login

# 2. Accept gated model licenses on HuggingFace:
#    - https://huggingface.co/black-forest-labs/FLUX.1-dev
#    - https://huggingface.co/black-forest-labs/FLUX.2-klein-9B (if using --flux2)

# 3. Download models
python download-models.py /path/to/models              # FLUX.1-dev only (~18GB)
python download-models.py /path/to/models --fill       # + FLUX.1-Fill-dev for outpainting (~22GB)
python download-models.py /path/to/models --checkpoints # + SDXL base (~6.9GB)
python download-models.py /path/to/models --flux2       # + FLUX.2-klein-9B (~17GB)
python download-models.py /path/to/models --kontext      # + FLUX.1-Kontext-dev (~22GB)
python download-models.py /path/to/models --hidream     # + HiDream-I1 Dev FP8 (~32GB)
python download-models.py /path/to/models --hidream-gguf # + HiDream-I1 Dev GGUF Q5_K_M (~13.5GB, fits 16GB VRAM)
python download-models.py /path/to/models --hidream-fast # + HiDream-I1 Fast FP8 (~17GB, 16 steps)
python download-models.py /path/to/models --hidream-o1  # + HiDream-O1-Image Dev MXFP8 (~8.9GB, self-contained)
python download-models.py /path/to/models --hidream-o1-base # + HiDream-O1-Image base MXFP8 (~8.9GB)
python download-models.py /path/to/models --qwen-image  # + Qwen-Image FP8 (~29GB)
python download-models.py /path/to/models --extras      # + embeddings, LoRAs, upscalers
python download-models.py /path/to/models --all         # everything (~160GB+)
python download-models.py --list                        # see all available downloads

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

## Supported Models

| Model | Type | Size | Loader | Use Case |
|-------|------|------|--------|----------|
| FLUX.1-dev (fp8) | UNet + CLIP + VAE | ~18GB | `UNETLoader` | High quality text-to-image, 12B params |
| FLUX.1-Fill-dev | UNet | ~22GB | `UNETLoader` | **Inpainting & outpainting** (purpose-built) |
| FLUX.2-klein-9B | UNet | ~17GB | `UNETLoader` | Faster FLUX, 9B params |
| FLUX.1-Kontext-dev | UNet | ~22GB | `UNETLoader` | Context-aware image editing |
| SDXL Base 1.0 | Checkpoint | ~6.9GB | `CheckpointLoaderSimple` | Mature ecosystem, LoRA support |
| HiDream-I1 Dev (fp8) | UNet + 4×CLIP + VAE | ~32GB | `UNETLoader` + `QuadrupleCLIPLoader` | 17B DiT, requires 4 text encoders |
| HiDream-I1 Dev (GGUF Q5_K_M) | UNet only¹ | ~13.5GB | `UnetLoaderGGUF` + `QuadrupleCLIPLoader` | Quantised HiDream — recommended for 16GB VRAM. Needs the [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) custom node |
| HiDream-I1 Fast (fp8) | UNet only¹ | ~17GB | `UNETLoader` + `QuadrupleCLIPLoader` | Distilled 16-step HiDream variant |
| HiDream-O1-Image Dev (mxfp8) | Checkpoint (all-in-one) | ~8.9GB | `CheckpointLoaderSimple` | 8B pixel-native unified model, 28 steps — **best quality-per-GB here for a 16GB Blackwell card** |
| HiDream-O1-Image base (mxfp8) | Checkpoint (all-in-one) | ~8.9GB | `CheckpointLoaderSimple` | Undistilled, 40–50 steps — marginal quality gain for ~2x the sampling time |
| Qwen-Image (fp8) | UNet + CLIP + VAE | ~29GB | `UNETLoader` + `CLIPLoader` | 20B MMDiT, excellent multilingual text |

All FLUX models share the same CLIP-L, T5-XXL, and VAE components. SDXL is a self-contained checkpoint. HiDream uses 4 dedicated text encoders (CLIP-G, CLIP-L, T5-XXL, Llama-3.1-8B). Qwen-Image uses a single Qwen 2.5 VL 7B encoder.

¹ The GGUF and Fast variants are diffusion models only — they reuse the text encoders and VAE from `--hidream`. Download that first, or the workflow will fail to load its encoders.

### Shared Model Directory

If you point `MODELS_PATH` at an existing model directory (e.g., one shared with `image-gen-cuda`), ComfyUI will automatically see all auxiliary files. The `extra_model_paths.yaml` maps the standard structure:

```
models/
├── checkpoints/    ← SD/SDXL .safetensors files (CheckpointLoaderSimple)
├── unet/           ← FLUX UNet / diffusion weights (UNETLoader)
├── clip/           ← Text encoders: CLIP-L, T5-XXL (DualCLIPLoader)
├── vae/            ← VAE / autoencoder (VAELoader)
├── loras/          ← LoRA fine-tunes (shared with image-gen-cuda)
├── embeddings/     ← Textual inversions (shared with image-gen-cuda)
└── upscaler/       ← Upscale models: UltraSharp, Nomos8k, RealESRGAN
```

**Note:** `image-gen-cuda` downloads models in HuggingFace diffusers format (directories like `sdxl-base/`). ComfyUI expects single `.safetensors` checkpoint files in `checkpoints/`. The diffusers-format models won't appear in ComfyUI, but all shared auxiliary files (LoRAs, embeddings, upscaler) work across both.

## Model Files

The `download-models.py` script fetches everything needed:

### FLUX.1-dev (default)

| File | Location | Size | Description |
|------|----------|------|-------------|
| `flux1-dev-fp8.safetensors` | `unet/` | ~12GB | FLUX.1-dev diffusion model (fp8 quantized) |
| `clip_l.safetensors` | `clip/` | ~250MB | CLIP-L text encoder |
| `t5xxl_fp8_e4m3fn.safetensors` | `clip/` | ~5GB | T5-XXL text encoder (fp8 quantized) |
| `ae.safetensors` | `vae/` | ~335MB | FLUX autoencoder / VAE |

### FLUX.1-Fill-dev (`--fill`)

| File | Location | Size | Description |
|------|----------|------|-------------|
| `flux1-fill-dev.safetensors` | `unet/` | ~22GB | Dedicated inpainting/outpainting model |

Purpose-built for inpainting and outpainting by Black Forest Labs. Far superior to using the base FLUX model for these tasks — supports full denoise (1.0) while maintaining consistency with the original image. Uses the same `clip/` and `vae/` components as FLUX.1-dev.

### FLUX.2-klein-9B (`--flux2`)

| File | Location | Size | Description |
|------|----------|------|-------------|
| `flux-2-klein-9b.safetensors` | `unet/` | ~17GB | FLUX.2-klein diffusion model |

Uses the same `clip/` and `vae/` components as FLUX.1-dev.

### SDXL Base 1.0 (`--checkpoints`)

| File | Location | Size | Description |
|------|----------|------|-------------|
| `sd_xl_base_1.0.safetensors` | `checkpoints/` | ~6.9GB | Complete SDXL checkpoint |

Self-contained — includes UNet, text encoders, and VAE in a single file.

### HiDream-O1-Image (`--hidream-o1`)

| File | Location | Size | Description |
|------|----------|------|-------------|
| `hidream_o1_image_dev_mxfp8.safetensors` | `checkpoints/` | ~8.9GB | HiDream-O1-Image Dev, 28 steps |

Add `--hidream-o1-base` for `hidream_o1_image_mxfp8.safetensors` (~8.9GB), the undistilled
50-step model.

Released May 2026 under the MIT licence, HiDream-O1-Image is a **pixel-native** ~8B unified
model, and *unified* here is meant in the strong sense. Its Qwen3-VL text backbone runs inside
`diffusion_model.*` on every sampling step, and its decoder is a pixel-space conversion VAE, so a
single file supplies MODEL, CLIP and VAE through `CheckpointLoaderSimple` — one ~8.9GB download,
against HiDream-I1's ~32GB of diffusion model plus four text encoders plus VAE.

> **Do not pair this with a Gemma-4 text encoder.** The ComfyUI tutorial lists
> `gemma4_e4b_it_fp8_scaled.safetensors` as a required text encoder, but that belongs to the
> official template's *optional prompt-refiner* branch — not to the model.
> `comfy/text_encoders/hidream_o1.py` describes itself as "tokenizer-only". Wiring a `CLIPLoader`
> into `CLIPTextEncode` fails at sampling time with
> `ValueError: not enough values to unpack (expected 4, got 1)`.

**Why mxfp8 and not fp8.** MXFP8 is block-scaled FP8 — one scale factor per 32 weights instead of
one per tensor — and Blackwell (RTX 50-series) implements it in hardware. On an RTX 5080 it is both
higher quality than plain `fp8_scaled` and roughly 30% faster than bf16. On Ada (RTX 40-series) it
must be dequantised on the fly and runs *slower*; swap the filename in `HIDREAM_O1_COMPONENTS` to
`hidream_o1_image_dev_fp8_scaled.safetensors` (~8.1GB) on that hardware.

**No GGUF variant is offered.** Unlike HiDream-I1, where GGUF is the only way to fit a 17B model in
16GB, the published HiDream-O1 GGUFs are Q6_K (~9.9GB) and Q8_0 (~11GB) — both *larger* than mxfp8
and without hardware acceleration. There is nothing to gain.

**VRAM.** At ~8.9GB the checkpoint fits the 5080's 16GB with room to spare — there is no
companion encoder to swap in and out, which is what makes this the most comfortable of the large
models here.

**Run it at 2048×2048.** This is the one setting that matters most on this model, and getting it
wrong is not a subtle loss. O1 patchifies raw RGB at `patch_size=32` with no VAE, so resolution
sets the token budget directly: 1024×1024 gives `(1024/32)² = 1024` tokens, a **quarter** of what
FLUX.2 gets at the same output size (VAE downscale 16 × patch 1 → 4096 tokens). It is also below
every bucket O1 was trained on — ~4MP: 2048×2048, 2304×1728, 2304×1792, 2496×1664, 2560×1440,
3104×1312 and their portrait transposes. ComfyUI's own `EmptyHiDreamO1LatentImage` help says
lower resolutions "go off-distribution and quality regresses noticeably", and defaults to 2048.

Generating at 1024×1024 is what makes O1 look like a weak SDXL rather than a model that
outscores FLUX.2 [Dev] on GenEval and DPG-Bench. Measured on an RTX 5080 at cfg 1.0, 28 steps:
2048×2048 takes **~28s** with the model resident and peaks around **11.7GB of 16GB** — so the
native resolution is comfortably affordable here. Reduce it only when VRAM forces you to, and
expect visibly softer, less detailed output when you do.

**Requires ComfyUI from 2026-05-12 or later** ([PR #13817](https://github.com/Comfy-Org/ComfyUI/pull/13817))
for the `EmptyHiDreamO1LatentImage`, `HiDreamO1ReferenceImages`, and `HiDreamO1PatchSeamSmoothing`
nodes. No custom extension is needed — rebuild with
`docker compose build --no-cache comfyui` to pull a new enough `master`. The `git clone` is a
cached Docker layer, so a plain `up -d --build` will happily reuse an old clone.

### Extras (`--extras`)

| File | Location | Size | Description |
|------|----------|------|-------------|
| `EasyNegative.safetensors` | `embeddings/` | ~25KB | Universal negative embedding (SD 1.5) |
| `bad-hands-5.pt` | `embeddings/` | ~25KB | Hand artifact fix (SD 1.5) |
| `negativeXL_D.safetensors` | `embeddings/` | ~10KB | Universal negative (SDXL) — manual download |
| `sdxl-lightning-4step.safetensors` | `loras/` | ~400MB | SDXL Lightning 4-step LoRA |
| `add-detail-xl.safetensors` | `loras/` | ~220MB | Add Detail XL — manual download |
| `detail-tweaker-xl.safetensors` | `loras/` | ~800MB | Detail Tweaker XL — manual download |
| `RealESRGAN_x4plus.pth` | `upscaler/` | ~64MB | Real-ESRGAN 4x baseline |
| `4x-UltraSharp.pth` | `upscaler/` | ~67MB | 4x UltraSharp — community favourite |
| `4xNomos8kSCHAT-L.safetensors` | `upscaler/` | ~316MB | Nomos8k HAT — extreme sharpness |

## Workflows

Pre-built workflow files in `workflows/`:

### FLUX.1-dev

| Workflow | Resolution | Steps | Use Case |
|----------|-----------|-------|----------|
| `flux1-dev-t2i.json` | 1024×1024 | 20 | Standard text-to-image |
| `flux1-dev-t2i-ultrawide.json` | 1344×576 | 25 | Ultrawide wallpaper (21:9) |
| `flux1-dev-t2i-ultrawide-upscaled.json` | 1344×576 → 4x | 25 | Ultrawide + 4x upscale |
| `flux1-dev-api-template.json` | Configurable | 20 | API automation template |

### FLUX.2-klein-9B

| Workflow | Resolution | Steps | Use Case |
|----------|-----------|-------|----------|
| `flux2-klein-t2i-api-template.json` | Configurable | 20 | API automation template |
| `flux2-klein-t2i-ultrawide-upscaled.json` | 1344×576 → 4x | 20 | Ultrawide + 4x upscale |

### SDXL

| Workflow | Resolution | Steps | Use Case |
|----------|-----------|-------|----------|
| `sdxl-t2i-api-template.json` | Configurable | 30 | API automation template |
| `sdxl-t2i-ultrawide-upscaled.json` | 1344×576 → 4x | 30 | Ultrawide + 4x upscale |

### HiDream-I1

Requires `--hidream` for the four text encoders and VAE.

| Workflow | Resolution | Steps | Use Case |
|----------|-----------|-------|----------|
| `hidream-i1-dev-t2i.json` | 1024×1024 | 50 | Standard text-to-image |
| `hidream-i1-dev-t2i-ultrawide.json` | 2560×1024 → 4x | 50 | Ultrawide + upscale |
| `hidream-i1-dev-t2i-api-template.json` | 1024×1024 | 28 | API automation template |

### HiDream-O1-Image

Requires `--hidream-o1`. The checkpoint is self-contained — there is no text encoder or VAE to
download.

| Workflow | Resolution | Steps | Use Case |
|----------|-----------|-------|----------|
| `hidream-o1-t2i-api-template.json` | 2048×2048 | 28 | API automation template |

This template uses the `SamplerCustom` chain rather than `KSampler`, because HiDream-O1 needs
`ModelNoiseScale` to set an absolute training noise scale: **7.6 for Dev, 8.0 for base**. Getting
that value wrong is the most likely cause of washed-out or noisy output. The rest tracks the
official *Dev* template: `SamplerLCM` (1.0 / 1.0 / 2.5), `normal` scheduler, and **cfg 1.0** — Dev
is guidance-distilled, so a higher cfg does not merely waste time, it overcooks the image. Measured
at 2048×2048 on a 16GB 5080, cfg 5.0 gave blown highlights, chromatic speckle across flat texture
and invented clutter, while running **~50× slower** (44–51 s/step against ~1.0 s/step, peaking
15.7GB of 16.3GB): ComfyUI only skips the uncond pass at exactly 1.0, and at 2048 the doubled batch
no longer fits. Point `ckpt_name` at
`hidream_o1_image_mxfp8.safetensors` and raise steps to 40–50 to use the base model instead — base
is not distilled, so also swap node 7 back to `KSamplerSelect` / `dpmpp_2m_sde_gpu` and set
cfg 5.0 and noise scale 8.0, matching the official Full template.

It also includes `HiDreamO1PatchSeamSmoothing`. A patch-based pixel transformer leaves a faint
rectangular seam grid across flat texture — wood grain, paper, out-of-focus background — and that
node runs extra offset passes over the last 20% of sampling and blends them, which visibly cleans
it up. The official Dev template omits this node and the Full template runs it with
`passes: ramp_2_4` / `blend: median`; the values here were the ones verified on an RTX 5080.
Rewire nodes 8 and 9 back to node 3 to skip it.

Timings on an RTX 5080 at the template's settings (cfg 1.0, 28 steps, seam smoothing on):
**2048×2048 in ~28s** with the model resident, plus roughly 180s on the first run to load the
checkpoint from disk.

Note there is no separate text encoder node — `CLIPTextEncode` takes its CLIP from the checkpoint
loader's second output. See the warning in the model files section above.

#### Dev vs Dev-2604

`--hidream-o1` fetches the original Dev checkpoint (8 May 2026). There is a second, later release —
**Dev-2604** (14 May 2026), tuned specifically for text-to-image, where the original Dev is the
general unified model that also covers editing and personalisation. Dev-2604 is the checkpoint
behind the "leading open-weights text-to-image model" coverage, so it is the one most reviews are
actually describing. `--hidream-o1-2604` fetches it.

Getting it into ComfyUI is the awkward part. **Comfy-Org has never packaged 2604**, and no scaled
quantisation of it exists on HuggingFace — only bf16/fp16, plus one unscaled fp8 that is not worth
using (raw `F8_E4M3` with no `weight_scale` tensors, which would cost more than 2604 gains and
confound any comparison). The registry therefore points at a third-party single-file bf16
repackaging, checked against Comfy-Org's own layout before being added: identical 758-tensor key
set, zero shape mismatches, and native unpadded vision dims of 4304 matching ComfyUI's built-in
default. It loads through `CheckpointLoaderSimple` unchanged — swap `ckpt_name` and nothing else.

Measured on an RTX 5080 at 2048×2048, 28 steps, against the Dev mxfp8 baseline:

| | Dev mxfp8 (~8.9GB) | Dev-2604 bf16 (~16.4GB) |
|---|---|---|
| First image, cold load | ~210s | ~61s |
| Subsequent images | ~27s | ~33–36s |
| Peak VRAM | ~11.7GB | ~13.9GB |

Despite exceeding the card's 16GB on paper, partial offload costs only about 25% — it is usable,
not the minutes-per-image that the file size suggests.

**Quality was a wash in testing, so the stack still defaults to the original Dev.** Across three
matched prompts at one seed, Dev held a clear lead on portrait skin texture, 2604 held a clear lead
on macro subject rendering, and typography split (Dev richer scene, 2604 cleaner subject). Three
caveats before treating that as settled: the sample is tiny; the workflow's `noise_scale`, sampler
and seam-smoothing values all come from the official *Dev* template and may simply be wrong for
2604, which has no official template; and 2604 is designed to be driven by its Prompt Agent
(`HiDream-ai/Prompt-Refine`, served over vLLM), which the published rankings almost certainly used
and which none of this testing did.

### Qwen-Image

Strong multilingual text rendering — the best choice when the image must contain legible words.

| Workflow | Resolution | Steps | Use Case |
|----------|-----------|-------|----------|
| `qwen-image-t2i.json` | 1024×1024 | 30 | Standard text-to-image |
| `qwen-image-t2i-ultrawide.json` | 2560×1024 → 4x | 30 | Ultrawide + upscale |
| `qwen-image-t2i-api-template.json` | 1024×1024 | 20 | API automation template |

### Outpainting / Inpainting

| Workflow | Model | Use Case |
|----------|-------|----------|
| `flux-fill-outpaint-api-template.json` | FLUX.1-Fill-dev | **Recommended.** Single-pass outpainting with DifferentialDiffusion |
| `flux-fill-inpaint-api-template.json` | FLUX.1-Fill-dev | Inpainting (replace masked regions) |
| `outpaint-multipass-sdxl-flux-upscale.json` | SDXL + FLUX Fill | **Premium pipeline.** 4-phase: SDXL fill → restore original → FLUX refine → 4x upscale |

#### Outpainting Recommendations

**Why previous outpainting attempts were terrible:** Most approaches use the base generation model with a mask, leading to hard seams, style/lighting inconsistency, and detail loss at boundaries. The solution is threefold:

1. **Use FLUX.1-Fill-dev** — a purpose-built inpainting/outpainting model from Black Forest Labs. Unlike base FLUX or SDXL, it's trained specifically to maintain consistency with existing image content while generating new regions. You can use full denoise strength (1.0) without losing the original.

2. **DifferentialDiffusion** — this node makes denoising strength vary based on a gradient mask. Instead of a hard boundary between "keep" and "regenerate", you get a smooth gradient that eliminates visible seams.

3. **InpaintModelConditioning** — feeds the existing image content directly into the model's conditioning, so it "sees" what's already there when generating new content.

**Single-pass (`flux-fill-outpaint`):** Use for most outpainting tasks. Fast, clean, one model load. Adjust `left`/`top`/`right`/`bottom` padding in the `ImagePadForOutpaint` node. `feathering` controls the gradient blend width (40+ recommended).

**Multi-pass (`outpaint-multipass-sdxl-flux-upscale`):** Use when you need maximum quality or are extending significantly. The 4-phase pipeline:
1. **SDXL** does the initial rough fill (good at generating plausible content from scratch)
2. **Original image composited back** with feathered mask (restores any SDXL artifacts in the original area)
3. **FLUX Fill** refines the transition zone (fixes seams, matches style/lighting)
4. **4x UltraSharp upscale** for final output

Both workflows require `flux1-fill-dev.safetensors` — download with `--fill` flag.

### Loading a Workflow

**Via UI:** Open ComfyUI → Load → select the `.json` file.

**Via API:** Use the api-template format:

```bash
# FLUX.1-dev example
curl -s http://server-name:8188/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": {
      "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-dev-fp8.safetensors", "weight_dtype": "fp8_e4m3fn"}},
      "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp8_e4m3fn.safetensors", "type": "flux"}},
      "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
      "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cyberpunk raven in neon rain", "clip": ["2", 0]}},
      "5": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
      "6": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["7", 0], "latent_image": ["5", 0], "seed": 0, "control_after_generate": "randomize", "steps": 20, "cfg": 3.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
      "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
      "8": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
      "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "flux1dev"}}
    }
  }'

# SDXL example
curl -s http://server-name:8188/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": {
      "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
      "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cyberpunk raven in neon rain, highly detailed", "clip": ["1", 1]}},
      "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "low quality, blurry, text, watermark", "clip": ["1", 1]}},
      "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
      "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0], "seed": 0, "control_after_generate": "randomize", "steps": 30, "cfg": 7.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0}},
      "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
      "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "sdxl"}}
    }
  }'
```

## Recommended Settings

### FLUX.1-dev / FLUX.2-klein

| Parameter | Value | Notes |
|-----------|-------|-------|
| Steps | 20–28 | 20 is good, 28 for max quality |
| CFG | 3.5 | FLUX uses guidance distillation; low CFG is correct |
| Sampler | `euler` | Best results with FLUX |
| Scheduler | `simple` | Standard for FLUX |
| Resolution | 1024×1024 | Native; also supports 1344×576 (21:9), 768×1344 (portrait) |
| Negative prompt | (empty) | FLUX does not use negative prompts |

### SDXL Base 1.0

| Parameter | Value | Notes |
|-----------|-------|-------|
| Steps | 25–40 | 30 is a good default |
| CFG | 6.0–8.0 | 7.5 is standard |
| Sampler | `dpmpp_2m` | Also good: `dpmpp_2m_sde`, `euler_ancestral` |
| Scheduler | `karras` | Best with DPM++ samplers |
| Resolution | 1024×1024 | Native; 1344×576 for ultrawide |
| Negative prompt | ✅ Use it | SDXL benefits strongly from negative prompts |

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

### FLUX.1-dev (fp8)

| Resolution | Steps | Expected Time | VRAM Usage |
|-----------|-------|--------------|------------|
| 1024×1024 | 20 | ~15–20s | ~13GB |
| 1024×1024 | 28 | ~20–30s | ~13GB |
| 1344×576 | 25 | ~15–20s | ~12GB |

### FLUX.2-klein-9B

| Resolution | Steps | Expected Time | VRAM Usage |
|-----------|-------|--------------|------------|
| 1024×1024 | 20 | ~10–15s | ~11GB |
| 1344×576 | 20 | ~10–15s | ~10GB |

### SDXL Base 1.0

| Resolution | Steps | Expected Time | VRAM Usage |
|-----------|-------|--------------|------------|
| 1024×1024 | 30 | ~8–12s | ~7GB |
| 1344×576 | 30 | ~6–10s | ~6GB |

First generation for each model includes model loading (~10–15s extra). Switching between models requires unloading the previous one.

## Comparison: XPU vs CUDA

| Feature | `comfyui` (Intel XPU) | `comfyui-cuda` (this) |
|---------|----------------------|----------------------|
| Target hardware | Intel Arc / iGPU | NVIDIA GPU |
| FLUX support | Limited | Full (fp8) |
| SDXL support | Limited | Full |
| Image format | Pre-built Docker image | Custom Dockerfile |
| Speed (FLUX.1-dev) | Very slow (CPU fallback) | ~15–20s (RTX 5080) |
| Speed (SDXL) | ~60s+ | ~8–12s (RTX 5080) |
| VRAM required | N/A (shared memory) | ~7–13GB depending on model |

## Troubleshooting

### sm_120 / Blackwell Warnings

If you see `sm_120 not compatible` warnings, the PyTorch build doesn't include Blackwell kernels. Rebuild the image to pick up the latest nightly:

```bash
docker compose build --no-cache
```

### Out of Memory

If you hit OOM errors:

1. Close other GPU-using applications
2. Check VRAM: `nvidia-smi`
3. Use fp8 quantized models where available
4. Switch to a smaller model (FLUX.2-klein uses less VRAM than FLUX.1-dev)

### Model Not Found

Ensure model files are in the correct subdirectories matching `extra_model_paths.yaml`. Run `python list-models.py /path/to/models` to verify.

### Switching Models

ComfyUI can only hold one large model in VRAM at a time. When you submit a workflow with a different model (e.g., SDXL after FLUX), ComfyUI will automatically unload the previous model and load the new one. This adds ~10–15s on the first generation after a switch.
