#!/usr/bin/env python3
"""
Download all models for the image-gen-cuda pipeline.

Usage:
    python download-models.py [MODELS_DIR]

    MODELS_DIR defaults to ./models if not specified.
    Set environment variable MODELS_PATH as an alternative.

Examples:
    python download-models.py D:\\SD\\models
    python download-models.py /data/models
    MODELS_PATH=/data/models python download-models.py

Requires: pip install diffusers transformers accelerate torch safetensors
Optional: pip install realesrgan basicsr gfpgan  (for upscaler)
"""

import os
import sys
import time
import shutil
import argparse
import urllib.request

# ─── Model Registry ───────────────────────────────────────────────────────────

MODELS = {
    # ── Core / Required ──────────────────────────────────────────────────────
    "sd-v1-5": {
        "hf_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "pipeline": "StableDiffusionPipeline",
        "description": "Stable Diffusion 1.5 — the workhorse",
        "size_approx": "~5GB",
        "required": True,
    },
    "sd15-inpainting": {
        "hf_id": "runwayml/stable-diffusion-inpainting",
        "pipeline": "StableDiffusionInpaintPipeline",
        "description": "SD 1.5 Inpainting — for outpainting/inpainting",
        "size_approx": "~5GB",
        "required": False,
    },

    # ── SD 1.5 Fine-tunes (high quality) ─────────────────────────────────────
    "dreamshaper-8": {
        "hf_id": "Lykon/dreamshaper-8",
        "pipeline": "StableDiffusionPipeline",
        "description": "DreamShaper 8 — photorealistic + fantasy, excellent all-rounder",
        "size_approx": "~5GB",
        "required": False,
    },
    "realistic-vision-6": {
        "hf_id": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
        "pipeline": "StableDiffusionPipeline",
        "description": "Realistic Vision v6 — best SD1.5 photorealism model",
        "size_approx": "~5GB",
        "required": False,
    },

    # ── SDXL Models ───────────────────────────────────────────────────────────
    "sdxl-turbo": {
        "hf_id": "stabilityai/sdxl-turbo",
        "pipeline": "StableDiffusionXLPipeline",
        "description": "SDXL Turbo — fast 4-step generation",
        "size_approx": "~7GB",
        "required": False,
    },
    "sdxl-base": {
        "hf_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline": "StableDiffusionXLPipeline",
        "description": "SDXL 1.0 Base — highest quality SDXL, use with refiner",
        "size_approx": "~7GB",
        "required": False,
    },
    "sdxl-refiner": {
        "hf_id": "stabilityai/stable-diffusion-xl-refiner-1.0",
        "pipeline": "StableDiffusionXLImg2ImgPipeline",
        "description": "SDXL Refiner — post-process SDXL base outputs for added detail",
        "size_approx": "~7GB",
        "required": False,
    },
    "juggernaut-xl": {
        "hf_id": "RunDiffusion/Juggernaut-XL-v9",
        "pipeline": "StableDiffusionXLPipeline",
        "description": "Juggernaut XL v9 — best SDXL photorealism, exceptional detail",
        "size_approx": "~7GB",
        "required": False,
    },
    "dreamshaper-xl": {
        "hf_id": "Lykon/dreamshaper-xl-1-0",
        "pipeline": "StableDiffusionXLPipeline",
        "description": "DreamShaper XL — SDXL fine-tune, versatile photorealistic+artistic",
        "size_approx": "~7GB",
        "required": False,
    },
    "realvis-xl-4": {
        "hf_id": "SG161222/RealVisXL_V4.0",
        "pipeline": "StableDiffusionXLPipeline",
        "description": "RealVisXL v4 — ultra-photorealistic SDXL, best for portraits+scenes",
        "size_approx": "~7GB",
        "required": False,
    },

    # ── FLUX Models (state of the art) ───────────────────────────────────────
    "flux-schnell": {
        "hf_id": "black-forest-labs/FLUX.1-schnell",
        "pipeline": "FluxPipeline",
        "description": "FLUX.1 Schnell — fast, state-of-the-art quality (Apache 2.0)",
        "size_approx": "~12GB",
        "required": False,
    },
    "flux-dev": {
        "hf_id": "black-forest-labs/FLUX.1-dev",
        "pipeline": "FluxPipeline",
        "description": "FLUX.1 Dev — highest quality, slower than schnell (non-commercial)",
        "size_approx": "~24GB",
        "required": False,
    },
}

UPSCALER = {
    "name": "RealESRGAN_x4plus",
    "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "path": "upscaler/RealESRGAN_x4plus.pth",
    "size_approx": "~64MB",
}

# LoRAs — lightweight fine-tunes that modify model output style/subject
# Place in models/loras/. Load at inference time alongside base model.
LORAS = {
    "detail-tweaker-xl": {
        "hf_id": "stabilityai/stable-diffusion-xl-base-1.0",  # placeholder
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
        "path": "loras/detail-tweaker-xl.safetensors",
        "description": "Detail Tweaker XL — adds micro-detail and sharpness to SDXL",
        "base_model": "sdxl",
        "size_approx": "~800MB",
        "hf_direct": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
        "note": "Download manually from CivitAI: https://civitai.com/models/122359",
    },
    "film-grain-xl": {
        "url": None,
        "path": "loras/film-grain-xl.safetensors",
        "description": "Film Grain XL — cinematic film grain for photorealistic images",
        "base_model": "sdxl",
        "size_approx": "~150MB",
        "note": "Download manually from CivitAI: https://civitai.com/models/202388",
    },
    "lightning-xl": {
        "url": "https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_4step_lora.safetensors",
        "path": "loras/sdxl-lightning-4step.safetensors",
        "description": "SDXL Lightning 4-step LoRA — fast high-quality generation (ByteDance)",
        "base_model": "sdxl",
        "size_approx": "~400MB",
    },
}

# Embeddings — textual inversion embeddings that add concepts/fix issues
# Place in models/embeddings/. Reference in prompt as <embedding_name>.
EMBEDDINGS = {
    "easy-negative": {
        "url": "https://huggingface.co/datasets/gsdf/EasyNegative/resolve/main/EasyNegative.safetensors",
        "path": "embeddings/EasyNegative.safetensors",
        "description": "EasyNegative — universal negative prompt embedding for SD1.5",
        "base_model": "sd15",
        "size_approx": "~25KB",
        "usage": "Add 'EasyNegative' to negative prompt",
    },
    "bad-hands-5": {
        "url": "https://huggingface.co/yesyeahvh/bad-hands-5/resolve/main/bad-hands-5.pt",
        "path": "embeddings/bad-hands-5.pt",
        "description": "bad-hands-5 — fixes hand generation artifacts in SD1.5",
        "base_model": "sd15",
        "size_approx": "~25KB",
        "usage": "Add 'bad-hands-5' to negative prompt",
    },
    "negative-xl": {
        "url": "https://huggingface.co/gsdf/Counterfeit-XL/resolve/main/embedding/negativeXL_D.safetensors",
        "path": "embeddings/negativeXL_D.safetensors",
        "description": "NegativeXL — universal negative embedding for SDXL models",
        "base_model": "sdxl",
        "size_approx": "~10KB",
        "usage": "Add 'negativeXL_D' to negative prompt",
    },
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_step(text):
    print(f"\n  → {text}")


def print_ok(text):
    print(f"  ✓ {text}")


def print_skip(text):
    print(f"  ⊘ {text}")


def print_fail(text):
    print(f"  ✗ {text}")


def dir_has_model(path):
    """Check if a directory looks like it contains a downloaded model."""
    if not os.path.isdir(path):
        return False
    # Look for typical diffusers model files
    markers = ["model_index.json", "scheduler", "unet", "config.json"]
    contents = os.listdir(path)
    return any(m in contents for m in markers)


def format_size(bytes_val):
    """Format bytes as human-readable."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}TB"


def get_dir_size(path):
    """Get total size of a directory."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


# ─── Download Functions ───────────────────────────────────────────────────────

def download_diffusers_model(hf_id, pipeline_class, save_path):
    """Download a model from HuggingFace and save locally."""
    import torch
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        StableDiffusionInpaintPipeline,
        FluxPipeline,
    )

    pipeline_map = {
        "StableDiffusionPipeline": StableDiffusionPipeline,
        "StableDiffusionXLPipeline": StableDiffusionXLPipeline,
        "StableDiffusionInpaintPipeline": StableDiffusionInpaintPipeline,
        "FluxPipeline": FluxPipeline,
    }

    cls = pipeline_map[pipeline_class]
    print_step(f"Downloading {hf_id}...")

    t0 = time.time()
    pipe = cls.from_pretrained(hf_id, torch_dtype=torch.float16)

    print_step(f"Saving to {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    pipe.save_pretrained(save_path)

    elapsed = time.time() - t0
    size = get_dir_size(save_path)
    print_ok(f"Saved ({format_size(size)}) in {elapsed:.0f}s")

    # Free memory
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


def download_upscaler(url, save_path):
    """Download the Real-ESRGAN upscaler model."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print_step(f"Downloading Real-ESRGAN from {url}...")
    t0 = time.time()

    urllib.request.urlretrieve(url, save_path)

    elapsed = time.time() - t0
    size = os.path.getsize(save_path)
    print_ok(f"Saved ({format_size(size)}) in {elapsed:.0f}s")


def download_file(url, save_path, label):
    """Download a single file (LoRA, embedding, etc.) from a URL."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print_step(f"Downloading {label} from {url}...")
    t0 = time.time()

    urllib.request.urlretrieve(url, save_path)

    elapsed = time.time() - t0
    size = os.path.getsize(save_path)
    print_ok(f"Saved ({format_size(size)}) in {elapsed:.0f}s")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download models for image-gen-cuda pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models downloaded:
  [required]  SD 1.5              (~5GB)   — default generation model
  [optional]  DreamShaper 8       (~5GB)   — high quality SD 1.5 variant
  [optional]  SDXL Turbo          (~7GB)   — fast SDXL generation
  [optional]  SDXL 1.0 Base       (~7GB)   — highest quality SD model
  [optional]  FLUX.1 Schnell      (~12GB)  — state-of-the-art quality
  [optional]  SD 1.5 Inpainting   (~5GB)   — for outpainting feature
  [optional]  Real-ESRGAN 4x      (~64MB)  — image upscaler

Total if all downloaded: ~41GB
Minimum (required only): ~5GB
        """,
    )
    parser.add_argument(
        "models_dir",
        nargs="?",
        default=os.environ.get("MODELS_PATH", "./models"),
        help="Directory to download models into (default: ./models or $MODELS_PATH)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download all models including optional ones",
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=list(MODELS.keys()) + ["upscaler"],
        help="Download specific models by key",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip models that already exist locally (default: true)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if model exists locally",
    )
    parser.add_argument(
        "--loras", action="store_true",
        help="Also download LoRA fine-tunes (included automatically with --all)",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_models",
        help="List available models, LoRAs, and embeddings, then exit",
    )

    args = parser.parse_args()

    # List mode
    if args.list_models:
        print_header("Available Models")
        for key, config in MODELS.items():
            req = "[required]" if config["required"] else "[optional]"
            print(f"  {req:10s}  {key:25s}  {config['size_approx']:>6s}  {config['description']}")
        print(f"\n  {'[optional]':10s}  {'upscaler':25s}  {UPSCALER['size_approx']:>6s}  Real-ESRGAN 4x upscaler")
        print_header("Available LoRAs")
        for key, config in LORAS.items():
            note = f" ⚠ {config['note']}" if config.get('note') else ""
            print(f"  [lora]      {key:25s}  {config['size_approx']:>6s}  {config['description']}{note}")
        print_header("Available Embeddings")
        for key, config in EMBEDDINGS.items():
            print(f"  [embed]     {key:25s}  {config['size_approx']:>6s}  {config['description']}")
        return

    models_dir = os.path.abspath(args.models_dir)
    print_header(f"Image Gen CUDA — Model Downloader")
    print(f"  Target directory: {models_dir}")

    os.makedirs(models_dir, exist_ok=True)

    # Determine which models to download
    if args.models:
        selected_model_keys = [k for k in args.models if k != "upscaler"]
        download_upscaler_flag = "upscaler" in args.models
    elif args.all:
        selected_model_keys = list(MODELS.keys())
        download_upscaler_flag = True
    else:
        # Default: required models + upscaler
        selected_model_keys = [k for k, v in MODELS.items() if v["required"]]
        download_upscaler_flag = True
        print(f"\n  Downloading required models only. Use --all for everything.")
        print(f"  Or --models sd15 dreamshaper-8 sdxl-turbo for specific ones.")

    # Download diffusers models
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for key in selected_model_keys:
        config = MODELS[key]
        save_path = os.path.join(models_dir, key)

        print_header(f"{key} — {config['description']}")
        print(f"  HuggingFace: {config['hf_id']}")
        print(f"  Size: {config['size_approx']}")

        if dir_has_model(save_path) and not args.force:
            size = get_dir_size(save_path)
            print_skip(f"Already exists ({format_size(size)})")
            total_skipped += 1
            continue

        try:
            download_diffusers_model(
                config["hf_id"],
                config["pipeline"],
                save_path,
            )
            total_downloaded += 1
        except Exception as e:
            print_fail(f"Failed: {e}")
            total_failed += 1

    # Download upscaler
    if download_upscaler_flag:
        upscaler_path = os.path.join(models_dir, UPSCALER["path"])
        print_header(f"Real-ESRGAN 4x Upscaler")

        if os.path.exists(upscaler_path) and not args.force:
            size = os.path.getsize(upscaler_path)
            print_skip(f"Already exists ({format_size(size)})")
            total_skipped += 1
        else:
            try:
                download_upscaler(UPSCALER["url"], upscaler_path)
                total_downloaded += 1
            except Exception as e:
                print_fail(f"Failed: {e}")
                total_failed += 1

    # Download embeddings (always, they're tiny)
    print_header("Embeddings (tiny, always downloaded)")
    for key, config in EMBEDDINGS.items():
        embed_path = os.path.join(models_dir, config["path"])
        if os.path.exists(embed_path) and not args.force:
            print_skip(f"{key} — already exists")
            total_skipped += 1
        elif config.get("url"):
            try:
                download_file(config["url"], embed_path, key)
                total_downloaded += 1
            except Exception as e:
                print_fail(f"{key}: {e}")
                total_failed += 1
        else:
            print_skip(f"{key} — no URL, manual download required")

    # Download LoRAs (only with --loras flag or --all)
    if args.all or getattr(args, 'loras', False):
        print_header("LoRAs")
        for key, config in LORAS.items():
            lora_path = os.path.join(models_dir, config["path"])
            if os.path.exists(lora_path) and not args.force:
                size = os.path.getsize(lora_path)
                print_skip(f"{key} — already exists ({format_size(size)})")
                total_skipped += 1
            elif config.get("url"):
                try:
                    download_file(config["url"], lora_path, key)
                    total_downloaded += 1
                except Exception as e:
                    print_fail(f"{key}: {e}")
                    total_failed += 1
            else:
                note = config.get("note", "no URL available")
                print_skip(f"{key} — manual download required: {note}")

    # Summary
    print_header("Summary")
    total_size = get_dir_size(models_dir)
    print(f"  Downloaded: {total_downloaded}")
    print(f"  Skipped:    {total_skipped}")
    print(f"  Failed:     {total_failed}")
    print(f"  Total size: {format_size(total_size)}")
    print(f"  Location:   {models_dir}")

    if total_downloaded > 0 or total_skipped > 0:
        print(f"\n  To use with docker compose:")
        print(f"    MODELS_PATH={models_dir} docker compose up -d --build")

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
