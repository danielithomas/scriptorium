#!/usr/bin/env python3
"""
Download models for ComfyUI (CUDA).

Supports FLUX.1-dev, FLUX.2-klein, SDXL, HiDream-I1, HiDream-I1 GGUF,
HiDream-I1 Fast, Qwen-Image, and auxiliary models (embeddings, LoRAs,
upscalers). ComfyUI uses separate component files for FLUX/HiDream/Qwen-Image
(UNet/diffusion, CLIP, VAE) and single-file checkpoints for SD/SDXL.

Usage:
    python download-models.py [MODELS_DIR]
    python download-models.py D:\\SD\\models
    python download-models.py /data/models --all
    python download-models.py --list
    python download-models.py /data/models --checkpoints    # SDXL checkpoint only
    python download-models.py /data/models --flux2          # Flux2-klein-9b
    python download-models.py /data/models --hidream        # HiDream-I1 Dev FP8
    python download-models.py /data/models --hidream-gguf   # HiDream-I1 Dev GGUF (Q5_K_M)
    python download-models.py /data/models --hidream-fast   # HiDream-I1 Fast FP8 (16 steps)
    python download-models.py /data/models --qwen-image     # Qwen-Image FP8

Requires: pip install huggingface-hub[cli]

For gated models (FLUX.1-dev, FLUX.2-klein), you must first:
    1. Accept the license on HuggingFace
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

# ─── FLUX.1-Fill-dev (Inpainting/Outpainting) ────────────────────────────────
# Purpose-built inpainting and outpainting model from Black Forest Labs.
# Far superior to using base FLUX for in/outpainting — supports full denoise
# (1.0) while maintaining consistency with the original image.
# Uses the same CLIP/T5/VAE components as FLUX.1-dev.

FLUX_FILL_COMPONENTS = {
    "flux1-fill-dev": {
        "hf_repo": "black-forest-labs/FLUX.1-Fill-dev",
        "hf_file": "flux1-fill-dev.safetensors",
        "subdir": "unet",
        "description": "FLUX.1-Fill-dev — dedicated inpainting/outpainting model",
        "size_approx": "~22GB",
        "required": True,
        "gated": True,
    },
}

# ─── FLUX.2 Models ───────────────────────────────────────────────────────────
# FLUX.2-klein is a 9B parameter model — smaller and faster than FLUX.1-dev.
# Uses the same CLIP/T5/VAE components as FLUX.1-dev.

FLUX2_COMPONENTS = {
    "flux2-klein-9b": {
        "hf_repo": "black-forest-labs/FLUX.2-klein-9B",
        "hf_file": "flux-2-klein-9b.safetensors",
        "subdir": "unet",
        "description": "FLUX.2-klein 9B diffusion model (smaller, faster than FLUX.1-dev)",
        "size_approx": "~17GB",
        "required": True,
        "gated": True,
    },
    "flux2-qwen3-8b": {
        "hf_repo": "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
        "hf_file": "split_files/text_encoders/qwen_3_8b.safetensors",
        "subdir": "clip",
        "description": "Qwen 3 8B text encoder for FLUX.2-klein-9B (replaces CLIP-L + T5-XXL)",
        "size_approx": "~16GB",
        "required": True,
        "gated": False,
    },
    "flux2-vae": {
        "hf_repo": "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
        "hf_file": "split_files/vae/flux2-vae.safetensors",
        "subdir": "vae",
        "description": "FLUX.2 VAE (different from FLUX.1 ae.safetensors)",
        "size_approx": "~335MB",
        "required": True,
        "gated": False,
    },
}

# ─── FLUX.1-Kontext (Image Editing) ──────────────────────────────────────────
# FLUX.1-Kontext-dev is a context-aware image editing model.
# Uses the same CLIP/T5/VAE components as FLUX.1-dev.

FLUX_KONTEXT_COMPONENTS = {
    "flux1-kontext-dev": {
        "hf_repo": "black-forest-labs/FLUX.1-Kontext-dev",
        "hf_file": "flux1-kontext-dev.safetensors",
        "subdir": "unet",
        "description": "FLUX.1-Kontext-dev — context-aware image editing model",
        "size_approx": "~22GB",
        "required": True,
        "gated": True,
    },
}

# ─── Checkpoints (Single-File Models) ────────────────────────────────────────
# Standard SD/SDXL models in single-file .safetensors format.
# These go into checkpoints/ and use CheckpointLoaderSimple in ComfyUI.

CHECKPOINTS = {
    "sdxl-base-1.0": {
        "hf_repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "hf_file": "sd_xl_base_1.0.safetensors",
        "subdir": "checkpoints",
        "description": "Stable Diffusion XL Base 1.0 — high quality, 1024px native",
        "size_approx": "~6.9GB",
        "gated": False,
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
        "url": None,
        "filename": "negativeXL_D.safetensors",
        "subdir": "embeddings",
        "description": "Universal negative embedding (SDXL)",
        "size_approx": "~10KB",
        "usage": "Add 'negativeXL_D' to negative prompt",
        "note": "Manual download from CivitAI: https://civitai.com/models/118418",
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
    "add-detail-xl": {
        "url": None,
        "filename": "add-detail-xl.safetensors",
        "subdir": "loras",
        "description": "Add Detail XL — enhances fine detail in SDXL generations",
        "size_approx": "~220MB",
        "note": "Manual download from CivitAI: https://civitai.com/models/82098",
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
        "filename": "cinematic style film grain style film noise style v1-step00001950.safetensors",
        "subdir": "loras",
        "description": "Cinematic film grain — film noise effect for cinematic look",
        "size_approx": "~300MB",
        "note": "Manual download from CivitAI: https://civitai.com/models/202388",
    },
}

# ─── HiDream-I1 Components ───────────────────────────────────────────────────
# HiDream-I1 is a 17B parameter diffusion transformer with four text encoders:
# CLIP-G, CLIP-L (both HiDream-specific variants), T5-XXL, and Llama-3.1-8B.
# The diffusion model goes into diffusion_models/ (mapped to unet/ by ComfyUI).
# All text encoders go into clip/ and are loaded via QuadrupleCLIPLoader.
# Source: https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI

HIDREAM_COMPONENTS = {
    "hidream-i1-dev-fp8": {
        "hf_repo": "Comfy-Org/HiDream-I1_ComfyUI",
        "hf_file": "split_files/diffusion_models/hidream_i1_dev_fp8.safetensors",
        "subdir": "unet",
        "description": "HiDream-I1 Dev diffusion model (FP8, ~16GB VRAM)",
        "size_approx": "~17GB",
        "required": True,
        "gated": False,
    },
    "hidream-clip-g": {
        "hf_repo": "Comfy-Org/HiDream-I1_ComfyUI",
        "hf_file": "split_files/text_encoders/clip_g_hidream.safetensors",
        "subdir": "clip",
        "description": "CLIP-G text encoder (HiDream-specific variant)",
        "size_approx": "~1.4GB",
        "required": True,
        "gated": False,
    },
    "hidream-clip-l": {
        "hf_repo": "Comfy-Org/HiDream-I1_ComfyUI",
        "hf_file": "split_files/text_encoders/clip_l_hidream.safetensors",
        "subdir": "clip",
        "description": "CLIP-L text encoder (HiDream-specific variant)",
        "size_approx": "~250MB",
        "required": True,
        "gated": False,
    },
    "hidream-t5xxl-fp8": {
        "hf_repo": "Comfy-Org/HiDream-I1_ComfyUI",
        "hf_file": "split_files/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors",
        "subdir": "clip",
        "description": "T5-XXL text encoder (FP8, scaled — HiDream variant)",
        "size_approx": "~5.2GB",
        "required": True,
        "gated": False,
    },
    "hidream-llama-3.1-8b-fp8": {
        "hf_repo": "Comfy-Org/HiDream-I1_ComfyUI",
        "hf_file": "split_files/text_encoders/llama_3.1_8b_instruct_fp8_scaled.safetensors",
        "subdir": "clip",
        "description": "Llama 3.1 8B Instruct text encoder (FP8) for HiDream",
        "size_approx": "~9GB",
        "required": True,
        "gated": False,
    },
    "hidream-vae": {
        "hf_repo": "Comfy-Org/HiDream-I1_ComfyUI",
        "hf_file": "split_files/vae/ae.safetensors",
        "subdir": "vae",
        "description": "HiDream VAE / autoencoder",
        "size_approx": "~335MB",
        "required": True,
        "gated": False,
    },
}

# ─── HiDream-I1 GGUF Components ──────────────────────────────────────────────
# GGUF-quantized diffusion models from city96/HiDream-I1-Dev-gguf (HuggingFace).
# Place model files in ComfyUI/models/diffusion_models/.
# Requires the ComfyUI-GGUF custom extension and the "Unet Loader (GGUF)" node
# (class_type: UnetLoaderGGUF). Uses the SAME text encoders and VAE as the FP8
# version in HIDREAM_COMPONENTS — download those separately with --hidream.
#
# Available quant sizes (all from city96/HiDream-I1-Dev-gguf):
#   Q4_K_M  ~11GB  — aggressive quant, fits 12GB VRAM
#   Q5_K_M  ~13.5GB — recommended for 16GB VRAM (quality/size balance)
#   Q6_K    ~14.7GB — high quality quant
#   Q8_0    ~17GB  — near-lossless, needs 24GB+ VRAM

HIDREAM_GGUF_COMPONENTS = {
    "hidream-i1-dev-gguf-q5km": {
        "hf_repo": "city96/HiDream-I1-Dev-gguf",
        "hf_file": "hidream-i1-dev-Q5_K_M.gguf",
        "subdir": "diffusion_models",
        "description": "HiDream-I1 Dev GGUF Q5_K_M — recommended for 16GB VRAM",
        "size_approx": "~13.5GB",
        "required": True,
        "gated": False,
    },
}

# ─── HiDream-I1 Fast Components ───────────────────────────────────────────────
# HiDream-I1 Fast is a distilled 16-step variant for quick iteration.
# Uses the SAME text encoders and VAE as HiDream-I1 Dev — download those with
# --hidream. Only the diffusion model differs.
# Source: https://huggingface.co/Comfy-Org/HiDream-I1_ComfyUI

HIDREAM_FAST_COMPONENTS = {
    "hidream-i1-fast-fp8": {
        "hf_repo": "Comfy-Org/HiDream-I1_ComfyUI",
        "hf_file": "split_files/diffusion_models/hidream_i1_fast_fp8.safetensors",
        "subdir": "unet",
        "description": "HiDream-I1 Fast diffusion model (FP8, 16 steps, quick iteration)",
        "size_approx": "~17GB",
        "required": True,
        "gated": False,
    },
}

# ─── Qwen-Image Components ────────────────────────────────────────────────────
# Qwen-Image is a 20B MMDiT model from Alibaba's Qwen team (Apache 2.0).
# Excellent multilingual text rendering and diverse artistic styles.
# Uses a single Qwen 2.5 VL 7B vision-language encoder (CLIPLoader, qwen_image type).
# Source: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI

QWEN_IMAGE_COMPONENTS = {
    "qwen-image-dit-fp8": {
        "hf_repo": "Comfy-Org/Qwen-Image_ComfyUI",
        "hf_file": "split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors",
        "subdir": "unet",
        "description": "Qwen-Image diffusion transformer (FP8, ~24GB VRAM)",
        "size_approx": "~20GB",
        "required": True,
        "gated": False,
    },
    "qwen-image-vl-encoder-fp8": {
        "hf_repo": "Comfy-Org/Qwen-Image_ComfyUI",
        "hf_file": "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "subdir": "clip",
        "description": "Qwen 2.5 VL 7B vision-language encoder (FP8)",
        "size_approx": "~9.4GB",
        "required": True,
        "gated": False,
    },
    "qwen-image-vae": {
        "hf_repo": "Comfy-Org/Qwen-Image_ComfyUI",
        "hf_file": "split_files/vae/qwen_image_vae.safetensors",
        "subdir": "vae",
        "description": "Qwen-Image VAE",
        "size_approx": "~254MB",
        "required": True,
        "gated": False,
    },
}

# ─── Upscalers ────────────────────────────────────────────────────────────────

UPSCALERS = {
    "realesrgan-x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
        "subdir": "upscaler",
        "description": "Real-ESRGAN 4x upscaler (solid baseline)",
        "size_approx": "~64MB",
    },
    "4x-ultrasharp": {
        "url": "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth",
        "filename": "4x-UltraSharp.pth",
        "subdir": "upscaler",
        "description": "4x UltraSharp — rich detail + texture, community favourite",
        "size_approx": "~67MB",
    },
    "4x-nomos8k-schat-l": {
        "url": "https://huggingface.co/Phhofm/4xNomos8kSCHAT-L/resolve/main/4xNomos8kSCHAT-L.pth",
        "filename": "4xNomos8kSCHAT-L.pth",
        "subdir": "upscaler",
        "description": "4x Nomos8kSCHAT-L (HAT) — photorealistic, extreme sharpness",
        "size_approx": "~316MB",
    },
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
    """Download a single file from HuggingFace using huggingface_hub.

    If filename contains subdirectories (e.g. 'split_files/vae/model.safetensors'),
    the file is downloaded and then moved to save_dir with just the basename.
    """
    # The final filename is just the basename (flatten nested HF paths)
    basename = os.path.basename(filename)
    save_path = os.path.join(save_dir, basename)

    if os.path.exists(save_path) and not force:
        size = os.path.getsize(save_path)
        print_skip(f"Already exists ({format_size(size)}): {basename}")
        return False

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print_fail("huggingface_hub not installed. Run: pip install huggingface-hub[cli]")
        return False

    os.makedirs(save_dir, exist_ok=True)
    print_step(f"Downloading {repo_id}/{filename}...")
    t0 = time.time()

    # Download to a temp directory first, then move to save_dir
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=tmp_dir,
            local_dir_use_symlinks=False,
        )
        # Move to final location (flatten any subdirectory structure)
        import shutil
        shutil.move(downloaded_path, save_path)

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


def download_registry(registry, models_dir, force=False, use_hf=True):
    """Download all items from a registry dict. Returns (downloaded, skipped, failed)."""
    downloaded = skipped = failed = 0

    for key, comp in registry.items():
        if use_hf and "hf_repo" in comp:
            save_dir = os.path.join(models_dir, comp["subdir"])
            # Use basename for checking existence (HF paths may be nested)
            basename = os.path.basename(comp["hf_file"])
            save_path = os.path.join(save_dir, basename)
            label = f"{comp['description']} ({comp['size_approx']})"
            print(f"\n  [{key}] {label}")

            if os.path.exists(save_path) and not force:
                size = os.path.getsize(save_path)
                print_skip(f"Already exists ({format_size(size)})")
                skipped += 1
                continue

            try:
                result = download_hf_file(
                    comp["hf_repo"], comp["hf_file"], save_dir, force=force
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

        elif "url" in comp:
            filename = comp.get("filename", comp.get("hf_file"))
            save_path = os.path.join(models_dir, comp["subdir"], filename)

            if comp["url"] is None:
                note = comp.get("note", "No URL — manual download required")
                print_skip(f"{key} — {note}")
                continue

            try:
                result = download_url(comp["url"], save_path, key, force=force)
                downloaded += 1 if result else 0
                skipped += 0 if result else 1
            except Exception as e:
                print_fail(f"{key}: {e}")
                failed += 1

    return downloaded, skipped, failed


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download models for ComfyUI CUDA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model groups:
  (default)         FLUX.1-dev components (~18GB)
  --checkpoints     SDXL base checkpoint (~6.9GB)
  --flux2           FLUX.2-klein-9B (~17GB)
  --kontext         FLUX.1-Kontext-dev (~22GB)
  --hidream         HiDream-I1 Dev FP8 + text encoders (~32GB)
  --hidream-gguf    HiDream-I1 Dev GGUF Q5_K_M (~13.5GB, needs text encoders)
  --hidream-fast    HiDream-I1 Fast FP8 (~17GB, 16 steps, needs text encoders)
  --qwen-image      Qwen-Image FP8 + VL encoder (~29GB)
  --extras          Embeddings, LoRAs, upscalers
  --all             Everything above

HiDream GGUF notes:
  Requires the ComfyUI-GGUF custom extension. Use "Unet Loader (GGUF)"
  (UnetLoaderGGUF) node instead of the standard UNETLoader. Text encoders
  and VAE are shared with --hidream; download those first. Other quant sizes:
  Q4_K_M (~11GB), Q6_K (~14.7GB), Q8_0 (~17GB) — swap the filename in the
  HIDREAM_GGUF_COMPONENTS registry as needed.

Model directory structure:
  models/
  ├── checkpoints/    SD/SDXL single-file checkpoints
  ├── unet/           FLUX/HiDream/Qwen-Image diffusion weights
  │                   (ComfyUI also maps diffusion_models/ → here)
  ├── clip/           Text encoders (CLIP-L, CLIP-G, T5-XXL, Llama, Qwen VL)
  ├── vae/            VAE / autoencoder
  ├── loras/          LoRA fine-tunes
  ├── embeddings/     Textual inversions
  └── upscaler/       Upscale models (RealESRGAN, UltraSharp, Nomos8k)

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
        help="Download everything: FLUX.1-dev + SDXL + FLUX.2 + Kontext + HiDream + HiDream GGUF + HiDream Fast + Qwen-Image + extras",
    )
    parser.add_argument(
        "--flux", action="store_true", default=True,
        help="Download FLUX.1-dev components (default: true)",
    )
    parser.add_argument(
        "--fill", action="store_true",
        help="Download FLUX.1-Fill-dev (inpainting/outpainting model)",
    )
    parser.add_argument(
        "--flux2", action="store_true",
        help="Download FLUX.2-klein-9B diffusion model",
    )
    parser.add_argument(
        "--kontext", action="store_true",
        help="Download FLUX.1-Kontext-dev (context-aware image editing)",
    )
    parser.add_argument(
        "--checkpoints", action="store_true",
        help="Download SDXL base checkpoint (single-file .safetensors)",
    )
    parser.add_argument(
        "--hidream", action="store_true",
        help="Download HiDream-I1 Dev (FP8) with all four text encoders (~32GB total)",
    )
    parser.add_argument(
        "--hidream-gguf", action="store_true", dest="hidream_gguf",
        help="Download HiDream-I1 Dev GGUF Q5_K_M diffusion model (~13.5GB, 16GB VRAM). "
             "Requires ComfyUI-GGUF extension. Text encoders/VAE shared with --hidream.",
    )
    parser.add_argument(
        "--hidream-fast", action="store_true", dest="hidream_fast",
        help="Download HiDream-I1 Fast FP8 diffusion model (~17GB, 16 steps). "
             "Text encoders/VAE shared with --hidream.",
    )
    parser.add_argument(
        "--qwen-image", action="store_true", dest="qwen_image",
        help="Download Qwen-Image (FP8) with Qwen 2.5 VL encoder (~29GB total)",
    )
    parser.add_argument(
        "--extras", action="store_true",
        help="Download embeddings, LoRAs, and upscalers (included with --all)",
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

        print_header("FLUX.1-Fill-dev (--fill or --all)")
        for key, c in FLUX_FILL_COMPONENTS.items():
            gated = " [gated]" if c.get("gated") else ""
            print(f"  {c['subdir'] + '/' + c['hf_file']:45s}  {c['size_approx']:>6s}  {c['description']}{gated}")

        print_header("FLUX.2-klein-9B (--flux2 or --all)")
        for key, c in FLUX2_COMPONENTS.items():
            gated = " [gated]" if c.get("gated") else ""
            print(f"  {c['subdir'] + '/' + c['hf_file']:45s}  {c['size_approx']:>6s}  {c['description']}{gated}")

        print_header("FLUX.1-Kontext-dev (--kontext or --all)")
        for key, c in FLUX_KONTEXT_COMPONENTS.items():
            gated = " [gated]" if c.get("gated") else ""
            print(f"  {c['subdir'] + '/' + c['hf_file']:45s}  {c['size_approx']:>6s}  {c['description']}{gated}")

        print_header("HiDream-I1 Dev FP8 (--hidream or --all)")
        for key, c in HIDREAM_COMPONENTS.items():
            gated = " [gated]" if c.get("gated") else ""
            print(f"  {c['subdir'] + '/' + os.path.basename(c['hf_file']):45s}  {c['size_approx']:>6s}  {c['description']}{gated}")

        print_header("HiDream-I1 Dev GGUF (--hidream-gguf or --all)")
        for key, c in HIDREAM_GGUF_COMPONENTS.items():
            gated = " [gated]" if c.get("gated") else ""
            print(f"  {c['subdir'] + '/' + os.path.basename(c['hf_file']):45s}  {c['size_approx']:>6s}  {c['description']}{gated}")

        print_header("HiDream-I1 Fast FP8 (--hidream-fast or --all)")
        for key, c in HIDREAM_FAST_COMPONENTS.items():
            gated = " [gated]" if c.get("gated") else ""
            print(f"  {c['subdir'] + '/' + os.path.basename(c['hf_file']):45s}  {c['size_approx']:>6s}  {c['description']}{gated}")

        print_header("Qwen-Image FP8 (--qwen-image or --all)")
        for key, c in QWEN_IMAGE_COMPONENTS.items():
            gated = " [gated]" if c.get("gated") else ""
            print(f"  {c['subdir'] + '/' + os.path.basename(c['hf_file']):45s}  {c['size_approx']:>6s}  {c['description']}{gated}")

        print_header("Checkpoints (--checkpoints or --all)")
        for key, c in CHECKPOINTS.items():
            gated = " [gated]" if c.get("gated") else ""
            print(f"  {c['subdir'] + '/' + c['hf_file']:45s}  {c['size_approx']:>6s}  {c['description']}{gated}")

        print_header("Embeddings (--extras or --all)")
        for key, c in EMBEDDINGS.items():
            note = f" ⚠ {c['note']}" if c.get("note") else ""
            print(f"  {c['subdir'] + '/' + c['filename']:45s}  {c['size_approx']:>6s}  {c['description']}{note}")

        print_header("LoRAs (--extras or --all)")
        for key, c in LORAS.items():
            note = f" ⚠ {c['note']}" if c.get("note") else ""
            print(f"  {c['subdir'] + '/' + c['filename']:45s}  {c['size_approx']:>6s}  {c['description']}{note}")

        print_header("Upscalers (--extras or --all)")
        for key, c in UPSCALERS.items():
            print(f"  {c['subdir'] + '/' + c['filename']:45s}  {c['size_approx']:>6s}  {c['description']}")
        return

    # ── Setup ─────────────────────────────────────────────────────────────────
    models_dir = os.path.abspath(args.models_dir)
    print_header("ComfyUI CUDA — Model Downloader")
    print(f"  Target directory: {models_dir}")
    os.makedirs(models_dir, exist_ok=True)

    # Ensure all subdirectories exist
    for subdir in ["checkpoints", "unet", "clip", "vae", "loras", "embeddings", "upscaler", "diffusion_models"]:
        os.makedirs(os.path.join(models_dir, subdir), exist_ok=True)

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    # ── FLUX.1-dev Components ─────────────────────────────────────────────────
    print_header("FLUX.1-dev Components")
    d, s, f = download_registry(FLUX_COMPONENTS, models_dir, force=args.force)
    total_downloaded += d; total_skipped += s; total_failed += f

    # ── FLUX.1-Fill-dev ─────────────────────────────────────────────────────
    if args.all or args.fill:
        print_header("FLUX.1-Fill-dev (Inpainting/Outpainting)")
        d, s, f = download_registry(FLUX_FILL_COMPONENTS, models_dir, force=args.force)
        total_downloaded += d; total_skipped += s; total_failed += f

    # ── FLUX.2-klein-9B ───────────────────────────────────────────────────────
    if args.all or args.flux2:
        print_header("FLUX.2-klein-9B")
        d, s, f = download_registry(FLUX2_COMPONENTS, models_dir, force=args.force)
        total_downloaded += d; total_skipped += s; total_failed += f

    # ── FLUX.1-Kontext-dev ────────────────────────────────────────────────────
    if args.all or args.kontext:
        print_header("FLUX.1-Kontext-dev")
        d, s, f = download_registry(FLUX_KONTEXT_COMPONENTS, models_dir, force=args.force)
        total_downloaded += d; total_skipped += s; total_failed += f

    # ── HiDream-I1 ────────────────────────────────────────────────────────────
    if args.all or args.hidream:
        print_header("HiDream-I1 Dev FP8")
        d, s, f = download_registry(HIDREAM_COMPONENTS, models_dir, force=args.force)
        total_downloaded += d; total_skipped += s; total_failed += f

    # ── HiDream-I1 GGUF ───────────────────────────────────────────────────────
    if args.all or args.hidream_gguf:
        print_header("HiDream-I1 Dev GGUF (Q5_K_M)")
        print("  Note: requires ComfyUI-GGUF extension. Text encoders/VAE: use --hidream.")
        d, s, f = download_registry(HIDREAM_GGUF_COMPONENTS, models_dir, force=args.force)
        total_downloaded += d; total_skipped += s; total_failed += f

    # ── HiDream-I1 Fast ───────────────────────────────────────────────────────
    if args.all or args.hidream_fast:
        print_header("HiDream-I1 Fast FP8")
        print("  Note: text encoders/VAE shared with --hidream. Download those first.")
        d, s, f = download_registry(HIDREAM_FAST_COMPONENTS, models_dir, force=args.force)
        total_downloaded += d; total_skipped += s; total_failed += f

    # ── Qwen-Image ────────────────────────────────────────────────────────────
    if args.all or args.qwen_image:
        print_header("Qwen-Image FP8")
        d, s, f = download_registry(QWEN_IMAGE_COMPONENTS, models_dir, force=args.force)
        total_downloaded += d; total_skipped += s; total_failed += f

    # ── Checkpoints (SDXL) ────────────────────────────────────────────────────
    if args.all or args.checkpoints:
        print_header("Checkpoints (SDXL)")
        d, s, f = download_registry(CHECKPOINTS, models_dir, force=args.force)
        total_downloaded += d; total_skipped += s; total_failed += f

    # ── Extras ────────────────────────────────────────────────────────────────
    if args.all or args.extras:
        print_header("Embeddings")
        d, s, f = download_registry(EMBEDDINGS, models_dir, force=args.force, use_hf=False)
        total_downloaded += d; total_skipped += s; total_failed += f

        print_header("LoRAs")
        d, s, f = download_registry(LORAS, models_dir, force=args.force, use_hf=False)
        total_downloaded += d; total_skipped += s; total_failed += f

        print_header("Upscalers")
        d, s, f = download_registry(UPSCALERS, models_dir, force=args.force, use_hf=False)
        total_downloaded += d; total_skipped += s; total_failed += f

    # ── Summary ───────────────────────────────────────────────────────────────
    print_header("Summary")
    total_size = get_dir_size(models_dir)
    print(f"  Downloaded: {total_downloaded}")
    print(f"  Skipped:    {total_skipped}")
    print(f"  Failed:     {total_failed}")
    print(f"  Total size: {format_size(total_size)}")
    print(f"  Location:   {models_dir}")

    if total_downloaded > 0 or total_skipped > 0:
        print(f"\n  Next steps:")
        print(f"    1. Set MODELS_PATH={models_dir} in .env")
        print(f"    2. docker compose up -d --build")
        print(f"    3. Open http://localhost:8188")

    # Show what wasn't downloaded
    not_downloaded = []
    if not args.all:
        if not args.fill:
            not_downloaded.append("--fill (FLUX.1-Fill-dev inpaint/outpaint, ~22GB)")
        if not args.flux2:
            not_downloaded.append("--flux2 (FLUX.2-klein-9B, ~17GB)")
        if not args.kontext:
            not_downloaded.append("--kontext (FLUX.1-Kontext-dev, ~22GB)")
        if not args.hidream:
            not_downloaded.append("--hidream (HiDream-I1 Dev FP8 + text encoders, ~32GB)")
        if not args.hidream_gguf:
            not_downloaded.append("--hidream-gguf (HiDream-I1 Dev GGUF Q5_K_M diffusion only, ~13.5GB)")
        if not args.hidream_fast:
            not_downloaded.append("--hidream-fast (HiDream-I1 Fast FP8 diffusion only, ~17GB)")
        if not args.qwen_image:
            not_downloaded.append("--qwen-image (Qwen-Image FP8 + VL encoder, ~29GB)")
        if not args.checkpoints:
            not_downloaded.append("--checkpoints (SDXL base, ~6.9GB)")
        if not args.extras:
            not_downloaded.append("--extras (embeddings, LoRAs, upscalers)")

    if not_downloaded:
        print(f"\n  Also available:")
        for item in not_downloaded:
            print(f"    {item}")
        print(f"\n  Or use --all to download everything.")

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
