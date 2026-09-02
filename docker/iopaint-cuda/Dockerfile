# Ubuntu 22.04 (Python 3.10), NOT 24.04 like the other CUDA stacks in this repo.
# IOPaint hard-pins Pillow==9.5.0, which has no cp312 wheel — on Python 3.12 pip
# falls back to compiling it from source and fails (RequiredDependencyException:
# zlib), and Pillow only gained 3.12 support in 10.1. Python 3.10 has a prebuilt
# wheel, so nothing is compiled.
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# IOPaint version to install (pip package). Override at build time to pin.
ARG IOPAINT_VERSION=1.6.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create venv to avoid PEP 668 issues
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA 12.8 support (required for Blackwell / RTX 50-series / sm_120).
# The upstream uiewy/iopaint image ships cu121, which starts fine but fails every
# inference with "no kernel image is available for execution on the device".
#
# Use the STABLE cu128 channel, not nightly: stable has shipped sm_120 since 2.7,
# and the nightly index rebuilds torch and torchvision independently, so a nightly
# pair pulled on different days fails to resolve against each other.
#
# Pinned to the combination verified end-to-end on an RTX 5080 (sm_120).
# Bump deliberately and re-run the smoke test in the README when you do.
ARG TORCH_VERSION=2.11.0+cu128
ARG TORCHVISION_VERSION=0.26.0+cu128
RUN pip install --no-cache-dir --retries 5 --timeout 120 \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url https://download.pytorch.org/whl/cu128

# Install IOPaint. Its dependency resolution can pull a CPU or cu121 torch over
# the top of ours, so pin the installed build as a constraint rather than
# discovering the downgrade at runtime.
RUN python3 -c "import torch, torchvision; print('torch==' + torch.__version__); print('torchvision==' + torchvision.__version__)" > /tmp/torch-constraint.txt \
    && cat /tmp/torch-constraint.txt \
    && pip install --no-cache-dir --retries 5 --timeout 120 -c /tmp/torch-constraint.txt \
       "iopaint${IOPAINT_VERSION:+==${IOPAINT_VERSION}}"

# Diffusers-format SD inpainting models need these to build their tokenizer.
# IOPaint's own dependencies do not pull them, so selecting an SD model in the UI
# fails at load time with "requires the protobuf library but it was not found",
# while the erase models (lama, migan, …) work fine — easy to miss.
RUN pip install --no-cache-dir --retries 5 --timeout 120 -c /tmp/torch-constraint.txt \
    protobuf sentencepiece

# Fail the build — not the first inpaint — if the cu128 build was replaced.
#
# This checks the wheel metadata, NOT torch.cuda.get_arch_list(): that call needs
# an initialisable CUDA runtime and returns [] during `docker build`, where no GPU
# is attached. The version string is what actually distinguishes a cu128 wheel from
# the CPU or cu121 one a dependency might drag in, and it needs no device.
RUN python3 -c "\
import torch, sys; \
v = torch.__version__; c = torch.version.cuda; \
print('torch:', v, '| cuda:', c); \
sys.exit(0 if '+cu128' in v and (c or '').startswith('12.8') \
         else 'FATAL: expected a cu128 build, got ' + v + ' (cuda ' + str(c) + ')')"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/')"

# Overridden by compose to apply IOPAINT_MODEL / IOPAINT_DEVICE / IOPAINT_EXTRA_ARGS.
CMD ["iopaint", "start", "--model=lama", "--device=cuda", "--host=0.0.0.0", "--port=8080"]
