#!/usr/bin/env bash
# Download FLUX.1-dev model components for ComfyUI
#
# FLUX.1-dev is a gated model — you MUST:
#   1. Accept the license at https://huggingface.co/black-forest-labs/FLUX.1-dev
#   2. Log in: huggingface-cli login
#
# Usage: ./download-flux1-dev.sh /path/to/models
#
# Creates the following structure:
#   models/
#   ├── unet/flux1-dev-fp8.safetensors        (~12GB, fp8 quantized)
#   ├── clip/clip_l.safetensors                (~250MB)
#   ├── clip/t5xxl_fp8_e4m3fn.safetensors      (~5GB, fp8 quantized)
#   └── vae/ae.safetensors                     (~335MB)

set -euo pipefail

MODELS_DIR="${1:?Usage: $0 /path/to/models}"

echo "=== FLUX.1-dev Model Downloader for ComfyUI ==="
echo "Target: ${MODELS_DIR}"
echo ""

# Check for huggingface-cli
if ! command -v huggingface-cli &>/dev/null; then
    echo "ERROR: huggingface-cli not found. Install with: pip install huggingface-hub[cli]"
    exit 1
fi

# Check authentication
if ! huggingface-cli whoami &>/dev/null; then
    echo "ERROR: Not logged in to Hugging Face. Run: huggingface-cli login"
    echo "You also need to accept the license at: https://huggingface.co/black-forest-labs/FLUX.1-dev"
    exit 1
fi

mkdir -p "${MODELS_DIR}"/{unet,clip,vae}

# --- FLUX.1-dev UNet (fp8 quantized — fits 16GB VRAM) ---
UNET_FILE="${MODELS_DIR}/unet/flux1-dev-fp8.safetensors"
if [ -f "${UNET_FILE}" ]; then
    echo "✓ UNet already exists: ${UNET_FILE}"
else
    echo "⬇ Downloading FLUX.1-dev UNet (fp8, ~12GB)..."
    huggingface-cli download Kijai/flux-fp8 flux1-dev-fp8.safetensors \
        --local-dir "${MODELS_DIR}/unet" \
        --local-dir-use-symlinks False
fi

# --- CLIP-L text encoder ---
CLIP_L_FILE="${MODELS_DIR}/clip/clip_l.safetensors"
if [ -f "${CLIP_L_FILE}" ]; then
    echo "✓ CLIP-L already exists: ${CLIP_L_FILE}"
else
    echo "⬇ Downloading CLIP-L (~250MB)..."
    huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors \
        --local-dir "${MODELS_DIR}/clip" \
        --local-dir-use-symlinks False
fi

# --- T5-XXL text encoder (fp8 quantized — saves ~20GB vs full) ---
T5_FILE="${MODELS_DIR}/clip/t5xxl_fp8_e4m3fn.safetensors"
if [ -f "${T5_FILE}" ]; then
    echo "✓ T5-XXL (fp8) already exists: ${T5_FILE}"
else
    echo "⬇ Downloading T5-XXL fp8 (~5GB)..."
    huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp8_e4m3fn.safetensors \
        --local-dir "${MODELS_DIR}/clip" \
        --local-dir-use-symlinks False
fi

# --- VAE (autoencoder) ---
VAE_FILE="${MODELS_DIR}/vae/ae.safetensors"
if [ -f "${VAE_FILE}" ]; then
    echo "✓ VAE already exists: ${VAE_FILE}"
else
    echo "⬇ Downloading FLUX VAE (~335MB)..."
    huggingface-cli download black-forest-labs/FLUX.1-dev ae.safetensors \
        --local-dir "${MODELS_DIR}/vae" \
        --local-dir-use-symlinks False
fi

echo ""
echo "=== Download complete ==="
echo ""
echo "Model directory structure:"
find "${MODELS_DIR}" -type f -name "*.safetensors" -exec ls -lh {} \;
echo ""
echo "Total disk usage:"
du -sh "${MODELS_DIR}"
echo ""
echo "Next steps:"
echo "  1. Set MODELS_PATH=${MODELS_DIR} in .env"
echo "  2. docker compose up -d --build"
echo "  3. Open http://localhost:8188"
echo "  4. Load the FLUX.1-dev workflow from workflows/flux1-dev-t2i.json"
