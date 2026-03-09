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
    "sd-v1-5": {
        "hf_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "pipeline": "StableDiffusionPipeline",
        "description": "Stable Diffusion 1.5 — the workhorse",
        "size_approx": "~5GB",
        "required": True,
    },
    "dreamshaper-8": {
        "hf_id": "Lykon/dreamshaper-8",
        "pipeline": "StableDiffusionPipeline",
        "description": "DreamShaper 8 — high quality, SD 1.5 based",
        "size_approx": "~5GB",
        "required": False,
    },
    "sdxl-turbo": {
        "hf_id": "stabilityai/sdxl-turbo",
        "pipeline": "StableDiffusionXLPipeline",
        "description": "SDXL Turbo — fast, 4 steps",
        "size_approx": "~7GB",
        "required": False,
    },
    "sdxl-base": {
        "hf_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline": "StableDiffusionXLPipeline",
        "description": "SDXL 1.0 Base — highest quality SD model",
        "size_approx": "~7GB",
        "required": False,
    },
    "flux-schnell": {
        "hf_id": "black-forest-labs/FLUX.1-schnell",
        "pipeline": "FluxPipeline",
        "description": "FLUX.1 Schnell — fast, excellent quality",
        "size_approx": "~12GB",
        "required": False,
    },
    "sd15-inpainting": {
        "hf_id": "runwayml/stable-diffusion-inpainting",
        "pipeline": "StableDiffusionInpaintPipeline",
        "description": "SD 1.5 Inpainting — for outpainting feature",
        "size_approx": "~5GB",
        "required": False,
    },
}

UPSCALER = {
    "name": "RealESRGAN_x4plus",
    "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "path": "upscaler/RealESRGAN_x4plus.pth",
    "size_approx": "~64MB",
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
        "--list", action="store_true", dest="list_models",
        help="List available models and exit",
    )

    args = parser.parse_args()

    # List mode
    if args.list_models:
        print_header("Available Models")
        for key, config in MODELS.items():
            req = "[required]" if config["required"] else "[optional]"
            print(f"  {req:10s}  {key:20s}  {config['size_approx']:>6s}  {config['description']}")
        print(f"  {'[optional]':10s}  {'upscaler':20s}  {UPSCALER['size_approx']:>6s}  Real-ESRGAN 4x upscaler")
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
