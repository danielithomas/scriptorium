#!/usr/bin/env python3
"""
iopaint-cuda model downloader.

Pre-caches models into the two host directories bind-mounted into the container:

    MODEL_DIR/torch/hub/checkpoints/   → /root/.cache/torch/hub/checkpoints/
    MODEL_DIR/huggingface/hub/         → /root/.cache/huggingface/hub/

Both paths are the defaults torch.hub and huggingface_hub read from inside the
container, so anything placed here is found on startup instead of re-downloaded.

By default this fetches only LaMa — the model the stack actually starts with
(IOPAINT_MODEL=lama). The diffusers inpainting models are several GB and only
used if you change IOPAINT_MODEL, so they are opt-in via --diffusers.

Run outside the container so the caches land on the host, not in a container layer.

Usage:
    export MODEL_DIR=/path/to/models
    python3 download-models.py                 # LaMa only (~200MB)
    python3 download-models.py --diffusers      # + diffusers inpainting models (~7GB)
    python3 download-models.py --force          # re-download even if present

Requires: pip install huggingface-hub   (only needed for --diffusers)
"""

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

# The status glyphs below are not encodable in cp1252, which is still the
# default console encoding on Windows — without this the script dies on its
# first print rather than on anything to do with models.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# --------------------------------------------------------------------------- #
# Configuration — edit the registries below to add or remove models.
# --------------------------------------------------------------------------- #

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models")).resolve()

# HF_CACHE_DIR / TORCH_CACHE_DIR mirror the compose overrides of the same name:
# set them to share caches with another stack, otherwise they derive from
# MODEL_DIR. Keep these in step with the volume mounts in compose.yaml.
HF_CACHE = Path(os.environ["HF_CACHE_DIR"]).resolve() if os.environ.get("HF_CACHE_DIR") \
    else MODEL_DIR / "huggingface"
TORCH_CACHE_ROOT = Path(os.environ["TORCH_CACHE_DIR"]).resolve() if os.environ.get("TORCH_CACHE_DIR") \
    else MODEL_DIR / "torch"

# huggingface_hub reads from $HF_HOME/hub — the "hub" segment is required, a
# file one level up is invisible to the library.
HF_HUB_CACHE = HF_CACHE / "hub"

# torch.hub.load_state_dict_from_url downloads into <torch hub dir>/checkpoints.
TORCH_CACHE = TORCH_CACHE_ROOT / "hub"
TORCH_CHECKPOINTS = TORCH_CACHE / "checkpoints"

# Single-file models IOPaint fetches by URL at runtime.
# The filename must match the URL basename — that is what IOPaint looks for.
TORCH_HUB_MODELS = {
    "lama": {
        "url": "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
        "filename": "big-lama.pt",
        "description": "LaMa inpainting — the stack's default model",
        "size_approx": "~196MB",
    },
}

# Diffusers-format inpainting models (--diffusers).
# Use canonical repo ids: the old runwayml/* paths only resolve through a
# HuggingFace rename redirect, and cache under the requested name either way.
DIFFUSERS_MODELS = {
    "stable-diffusion-v1-5/stable-diffusion-inpainting": {
        "description": "Stable Diffusion 1.5 Inpainting",
        "size_approx": "~4GB",
    },
    "Sanster/Realistic_Vision_V1.4-inpainting": {
        "description": "Realistic Vision v1.4 Inpainting",
        "size_approx": "~2GB",
    },
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

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


def ensure_dirs():
    for d in (HF_HUB_CACHE, TORCH_CHECKPOINTS):
        d.mkdir(parents=True, exist_ok=True)


def hf_repo_folder(repo_id: str) -> Path:
    """The directory huggingface_hub creates for a repo inside the hub cache."""
    return HF_HUB_CACHE / f"models--{repo_id.replace('/', '--')}"


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #

def download_torch_hub_model(key: str, spec: dict, force: bool) -> bool:
    save_path = TORCH_CHECKPOINTS / spec["filename"]
    label = f"{spec['description']} ({spec['size_approx']})"
    print(f"\n  [{key}] {label}")

    if save_path.exists() and not force:
        print_skip(f"Already exists ({format_size(save_path.stat().st_size)})")
        return False

    print_step(f"Downloading {spec['url']}…")
    t0 = time.time()
    tmp_path = save_path.with_suffix(save_path.suffix + ".part")
    try:
        urllib.request.urlretrieve(spec["url"], tmp_path)
        tmp_path.replace(save_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    print_ok(f"Saved ({format_size(save_path.stat().st_size)}) in {time.time() - t0:.0f}s")
    return True


def download_diffusers_model(repo_id: str, spec: dict, force: bool) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print_fail("huggingface_hub not installed. Run: pip install huggingface-hub")
        return False

    label = f"{spec['description']} ({spec['size_approx']})"
    print(f"\n  [{repo_id}] {label}")

    folder = hf_repo_folder(repo_id)
    if folder.exists() and not force:
        print_skip(f"Already cached: {folder}")
        return False

    print_step(f"Downloading {repo_id}…")
    t0 = time.time()
    snapshot_download(repo_id, cache_dir=str(HF_HUB_CACHE))
    print_ok(f"Done in {time.time() - t0:.0f}s → {folder}")
    return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Download models for iopaint-cuda",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model groups:
  (default)     LaMa — the model the stack starts with (~196MB)
  --diffusers   Diffusers inpainting models, only used if you change
                IOPAINT_MODEL away from lama (~7GB)

MODEL_DIR is read from the environment (or --model-dir) and must match the
value passed to docker compose.
""",
    )
    parser.add_argument(
        "--model-dir", default=None,
        help="Model cache directory (default: $MODEL_DIR, else ./models)",
    )
    parser.add_argument(
        "--diffusers", action="store_true",
        help="Also download the diffusers inpainting models (~7GB)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if the model already exists locally",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_models",
        help="List available downloads, then exit",
    )
    args = parser.parse_args()

    if args.model_dir:
        global MODEL_DIR, HF_CACHE, HF_HUB_CACHE, TORCH_CACHE_ROOT, TORCH_CACHE, TORCH_CHECKPOINTS
        MODEL_DIR = Path(args.model_dir).resolve()
        # --model-dir only overrides caches that weren't pinned explicitly.
        if not os.environ.get("HF_CACHE_DIR"):
            HF_CACHE = MODEL_DIR / "huggingface"
            HF_HUB_CACHE = HF_CACHE / "hub"
        if not os.environ.get("TORCH_CACHE_DIR"):
            TORCH_CACHE_ROOT = MODEL_DIR / "torch"
            TORCH_CACHE = TORCH_CACHE_ROOT / "hub"
            TORCH_CHECKPOINTS = TORCH_CACHE / "checkpoints"

    if args.list_models:
        print_header("Torch hub models (default)")
        for key, spec in TORCH_HUB_MODELS.items():
            print(f"  {key:12s}  {spec['size_approx']:>8s}  {spec['description']}")
        print_header("Diffusers models (--diffusers)")
        for repo_id, spec in DIFFUSERS_MODELS.items():
            print(f"  {repo_id:52s}  {spec['size_approx']:>8s}  {spec['description']}")
        return 0

    print_header("iopaint-cuda — Model Downloader")
    print(f"  MODEL_DIR:   {MODEL_DIR}")
    print(f"  Torch cache: {TORCH_CHECKPOINTS}")
    print(f"  HF cache:    {HF_HUB_CACHE}")
    ensure_dirs()

    downloaded = skipped = failed = 0

    print_header("Torch hub models")
    for key, spec in TORCH_HUB_MODELS.items():
        try:
            if download_torch_hub_model(key, spec, args.force):
                downloaded += 1
            else:
                skipped += 1
        except Exception as exc:
            print_fail(f"{key}: {exc}")
            failed += 1

    if args.diffusers:
        print_header("Diffusers models")
        for repo_id, spec in DIFFUSERS_MODELS.items():
            try:
                if download_diffusers_model(repo_id, spec, args.force):
                    downloaded += 1
                else:
                    skipped += 1
            except Exception as exc:
                print_fail(f"{repo_id}: {exc}")
                failed += 1
    else:
        print_header("Diffusers models")
        print_skip("Not requested — add --diffusers to fetch them (~7GB)")

    print_header("Summary")
    print(f"  Downloaded: {downloaded}    Skipped: {skipped}    Failed: {failed}")
    # Echo back the configuration actually used — in shared-cache mode MODEL_DIR
    # is unset and printing it would name a directory nothing was written to.
    print("\n  To run iopaint with the caches just populated:")
    if os.environ.get("HF_CACHE_DIR") or os.environ.get("TORCH_CACHE_DIR"):
        print(f"    HF_CACHE_DIR={HF_CACHE} TORCH_CACHE_DIR={TORCH_CACHE_ROOT} docker compose up -d")
    else:
        print(f"    MODEL_DIR={MODEL_DIR} docker compose up -d")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
