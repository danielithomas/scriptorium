#!/usr/bin/env python3
"""
Download FLUX.1-dev and other models for ComfyUI (CUDA).

ComfyUI uses separate component files for FLUX (UNet, CLIP, T5, VAE)
rather than monolithic checkpoints. This script handles both FLUX components
and standard checkpoint/LoRA/embedding downloads.

Usage:
    python download-models.py [MODELS_DIR]
    python download-models.py D:\\SD\\models
    python download-models.py /data/models --all
    python download-models.py --list

Requires: pip install huggingface-hub[cli]

For gated models (FLUX.1-dev), you must first:
    1. Accept the license at https://huggingface.co/black-forest-labs/FLUX.1-dev
    2. Log in: huggingface-cli login
"""

import os
import sys
import time
import argparse
import urllib.request

# ─── FLUX Component Registry ─────────────────────────────────────────────────
# FLUX models are loaded as separate components in ComfyUI, not as a single
# checkpoint. Each component goes into a different model subdirectory.

FLUX_COMPONENTS = {
    "flux1-dev-unet": {
        "hf_repo": "Kijai/flux-fp8",
        "hf_file": "flux1-dev-fp8.safetensors",
        "subdir": "unet",
        "description": "FLUX.1-dev diffusion model (fp8 quantized, fits 16GB VRAM)",
        "size_approx": "~12GB",
        "required": True,
        "gated": False,
    },
    "flux1-clip-l": {
        "hf_repo": "comfyanonymous/flux_text_encoders",
        "hf_file": "clip_l.safetensors",
        "subdir": "clip",
        "description": "CLIP-L text encoder",
        "size_approx": "~250MB",
        "required": True,
        "gated": False,
    },
    "flux1-t5xxl-fp8": {
        "hf_repo": "comfyanonymous/flux_text_encoders",
        "hf_file": "t5xxl_fp8_e4m3fn.safetensors",
        "subdir": "clip",
        "description": "T5-XXL text encoder (fp8 quantized, saves ~20GB vs full)",
        "size_approx": "~5GB",
        "required": True,
        "gated": False,
    },
    "flux1-vae": {
        "hf_repo": "black-forest-labs/FLUX.1-dev",
        "hf_file": "ae.safetensors",
        "subdir": "vae",
        "description": "FLUX autoencoder / VAE",
        "size_approx": "~335MB",
        "required": True,
        "gated": True,
    },
}

# ─── Embeddings ───────────────────────────────────────────────────────────────

EMBEDDINGS = {
    "easy-negative": {
        "url": "https://huggingface.co/datasets/gsdf/EasyNegative/resolve/main/EasyNegative.safetensors",
        "filename": "EasyNegative.safetensors",
        "subdir": "embeddings",
        "description": "Universal negative prompt embedding (SD 1.5)",
        "size_approx": "~25KB",
        "usage": "Add 'EasyNegative' to negative prompt",
    },
    "bad-hands-5": {
        "url": "https://huggingface.co/yesyeahvh/bad-hands-5/resolve/main/bad-hands-5.pt",
        "filename": "bad-hands-5.pt",
        "subdir": "embeddings",
        "description": "Fixes hand generation artifacts (SD 1.5)",
        "size_approx": "~25KB",
        "usage": "Add 'bad-hands-5' to negative prompt",
    },
    "negative-xl": {
        "url": "https://huggingface.co/gsdf/Counterfeit-XL/resolve/main/embedding/negativeXL_D.safetensors",
        "filename": "negativeXL_D.safetensors",
        "subdir": "embeddings",
        "description": "Universal negative embedding (SDXL)",
        "size_approx": "~10KB",
        "usage": "Add 'negativeXL_D' to negative prompt",
    },
}

# ─── LoRAs ────────────────────────────────────────────────────────────────────

LORAS = {
    "lightning-xl": {
        "url": "https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_4step_lora.safetensors",
        "filename": "sdxl-lightning-4step.safetensors",
        "subdir": "loras",
        "description": "SDXL Lightning 4-step LoRA — fast generation (ByteDance)",
        "size_approx": "~400MB",
    },
    "detail-tweaker-xl": {
        "url": None,
        "filename": "detail-tweaker-xl.safetensors",
        "subdir": "loras",
        "description": "Detail Tweaker XL — micro-detail and sharpness",
        "size_approx": "~800MB",
        "note": "Manual download from CivitAI: https://civitai.com/models/122359",
    },
    "film-grain-xl": {
        "url": None,
        "filename": "film-grain-xl.safetensors",
        "subdir": "loras",
        "description": "Film Grain XL — cinematic film grain effect",
        "size_approx": "~150MB",
        "note": "Manual download from CivitAI: https://civitai.com/models/202388",
    },
}

# ─── Upscaler ─────────────────────────────────────────────────────────────────

UPSCALER = {
    "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "filename": "RealESRGAN_x4plus.pth",
    "subdir": "upscaler",
    "description": "Real-ESRGAN 4x upscaler",
    "size_approx": "~64MB",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def print_header(text):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_step(text):
    print(f"\n  → {text}")


def print_ok(text):
    print(f"  ✓ {text}")


def print_skip(text):
    print(f"  ⊘ {text}")


def print_fail(text):
    print(f"  ✗ {text}")


def format_size(bytes_val):
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}TB"


def get_dir_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


# ─── Download Functions ───────────────────────────────────────────────────────

def download_hf_file(repo_id, filename, save_dir, force=False):
    """Download a single file from HuggingFace using huggingface_hub."""
    save_path = os.path.join(save_dir, filename)

    if os.path.exists(save_path) and not force:
        size = os.path.getsize(save_path)
        print_skip(f"Already exists ({format_size(size)}): {filename}")
        return False

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print_fail("huggingface_hub not installed. Run: pip install huggingface-hub[cli]")
        return False

    os.makedirs(save_dir, exist_ok=True)
    print_step(f"Downloading {repo_id}/{filename}...")
    t0 = time.time()

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=save_dir,
        local_dir_use_symlinks=False,
    )

    elapsed = time.time() - t0
    size = os.path.getsize(save_path)
    print_ok(f"Saved ({format_size(size)}) in {elapsed:.0f}s")
    return True


def download_url(url, save_path, label, force=False):
    """Download a file from a direct URL."""
    if os.path.exists(save_path) and not force:
        size = os.path.getsize(save_path)
        print_skip(f"Already exists ({format_size(size)}): {os.path.basename(save_path)}")
        return False

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print_step(f"Downloading {label}...")
    t0 = time.time()

    urllib.request.urlretrieve(url, save_path)

    elapsed = time.time() - t0
    size = os.path.getsize(save_path)
    print_ok(f"Saved ({format_size(size)}) in {elapsed:.0f}s")
    return True


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download models for ComfyUI CUDA (FLUX.1-dev + extras)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FLUX.1-dev components (default):
  unet/flux1-dev-fp8.safetensors           (~12GB)  fp8 quantized diffusion model
  clip/clip_l.safetensors                  (~250MB)  CLIP-L text encoder
  clip/t5xxl_fp8_e4m3fn.safetensors        (~5GB)   T5-XXL fp8 text encoder
  vae/ae.safetensors                       (~335MB)  FLUX autoencoder

Total FLUX.1-dev: ~18GB

With --all: also downloads embeddings, LoRAs, and upscaler.

Model directory structure:
  models/
  ├── unet/           FLUX UNet weights
  ├── clip/           Text encoders (CLIP-L, T5-XXL)
  ├── vae/            VAE / autoencoder
  ├── checkpoints/    Full model checkpoints (SD, SDXL .safetensors)
  ├── loras/          LoRA fine-tunes
  ├── embeddings/     Textual inversions
  └── upscaler/       Real-ESRGAN weights

This structure is shared with image-gen-cuda. If you point both stacks
at the same MODELS_PATH, LoRAs, embeddings, and upscaler files are
shared automatically.
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
        help="Download everything: FLUX components + embeddings + LoRAs + upscaler",
    )
    parser.add_argument(
        "--flux", action="store_true", default=True,
        help="Download FLUX.1-dev components (default: true)",
    )
    parser.add_argument(
        "--extras", action="store_true",
        help="Download embeddings, LoRAs, and upscaler (included with --all)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if files exist locally",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_models",
        help="List all available downloads, then exit",
    )

    args = parser.parse_args()

    # ── List mode ─────────────────────────────────────────────────────────────
    if args.list_models:
        print_header("FLUX.1-dev Components (default)")
        for key, c in FLUX_COMPONENTS.items():
            gated = " [gated]" if c.get("gated") else ""
            print(f"  {c['subdir'] + '/' + c['hf_file']:45s}  {c['size_approx']:>6s}  {c['description']}{gated}")

        print_header("Embeddings (--extras or --all)")
        for key, c in EMBEDDINGS.items():
            print(f"  {c['subdir'] + '/' + c['filename']:45s}  {c['size_approx']:>6s}  {c['description']}")

        print_header("LoRAs (--extras or --all)")
        for key, c in LORAS.items():
            note = f" ⚠ {c['note']}" if c.get("note") else ""
            print(f"  {c['subdir'] + '/' + c['filename']:45s}  {c['size_approx']:>6s}  {c['description']}{note}")

        print_header("Upscaler (--extras or --all)")
        u = UPSCALER
        print(f"  {u['subdir'] + '/' + u['filename']:45s}  {u['size_approx']:>6s}  {u['description']}")
        return

    # ── Setup ─────────────────────────────────────────────────────────────────
    models_dir = os.path.abspath(args.models_dir)
    print_header("ComfyUI CUDA — Model Downloader")
    print(f"  Target directory: {models_dir}")
    os.makedirs(models_dir, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0

    # ── FLUX.1-dev Components ─────────────────────────────────────────────────
    print_header("FLUX.1-dev Components")

    for key, comp in FLUX_COMPONENTS.items():
        save_dir = os.path.join(models_dir, comp["subdir"])
        save_path = os.path.join(save_dir, comp["hf_file"])

        print(f"\n  [{key}] {comp['description']} ({comp['size_approx']})")

        if os.path.exists(save_path) and not args.force:
            size = os.path.getsize(save_path)
            print_skip(f"Already exists ({format_size(size)})")
            skipped += 1
            continue

        try:
            result = download_hf_file(
                comp["hf_repo"], comp["hf_file"], save_dir, force=args.force
            )
            downloaded += 1 if result else 0
            skipped += 0 if result else 1
        except Exception as e:
            print_fail(f"Failed: {e}")
            if comp.get("gated"):
                print(f"         This is a gated model. Make sure you've:")
                print(f"         1. Accepted the license on HuggingFace")
                print(f"         2. Logged in: huggingface-cli login")
            failed += 1

    # ── Extras ────────────────────────────────────────────────────────────────
    if args.all or args.extras:
        # Embeddings
        print_header("Embeddings")
        for key, emb in EMBEDDINGS.items():
            save_path = os.path.join(models_dir, emb["subdir"], emb["filename"])
            try:
                result = download_url(emb["url"], save_path, key, force=args.force)
                downloaded += 1 if result else 0
                skipped += 0 if result else 1
            except Exception as e:
                print_fail(f"{key}: {e}")
                failed += 1

        # LoRAs
        print_header("LoRAs")
        for key, lora in LORAS.items():
            save_path = os.path.join(models_dir, lora["subdir"], lora["filename"])
            if lora.get("url"):
                try:
                    result = download_url(lora["url"], save_path, key, force=args.force)
                    downloaded += 1 if result else 0
                    skipped += 0 if result else 1
                except Exception as e:
                    print_fail(f"{key}: {e}")
                    failed += 1
            else:
                note = lora.get("note", "No URL — manual download required")
                print_skip(f"{key} — {note}")

        # Upscaler
        print_header("Upscaler")
        up_path = os.path.join(models_dir, UPSCALER["subdir"], UPSCALER["filename"])
        try:
            result = download_url(UPSCALER["url"], up_path, "Real-ESRGAN 4x", force=args.force)
            downloaded += 1 if result else 0
            skipped += 0 if result else 1
        except Exception as e:
            print_fail(f"Upscaler: {e}")
            failed += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print_header("Summary")
    total_size = get_dir_size(models_dir)
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped:    {skipped}")
    print(f"  Failed:     {failed}")
    print(f"  Total size: {format_size(total_size)}")
    print(f"  Location:   {models_dir}")

    if downloaded > 0 or skipped > 0:
        print(f"\n  Next steps:")
        print(f"    1. Set MODELS_PATH={models_dir} in .env")
        print(f"    2. docker compose up -d --build")
        print(f"    3. Open http://localhost:8188")
        print(f"    4. Load workflows/flux1-dev-t2i.json")

    if not args.all and not args.extras:
        print(f"\n  Tip: Run with --all to also download embeddings, LoRAs, and upscaler.")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
