#!/usr/bin/env python3
"""
List all model files in a ComfyUI models directory.

Scans the directory tree and reports every model file found,
grouped by subdirectory, with file sizes.

Usage:
    python list-models.py [MODELS_DIR]
    python list-models.py D:\\SD\\models
    python list-models.py /data/models
"""

import os
import sys

MODEL_EXTENSIONS = {
    ".safetensors", ".pth", ".pt", ".ckpt", ".bin", ".onnx", ".gguf",
}


def format_size(bytes_val):
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}TB"


def main():
    models_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODELS_PATH", "./models")
    models_dir = os.path.abspath(models_dir)

    if not os.path.isdir(models_dir):
        print(f"Error: {models_dir} is not a directory")
        sys.exit(1)

    print(f"Models directory: {models_dir}\n")

    # Collect files grouped by subdirectory
    groups = {}
    total_size = 0
    total_count = 0

    for dirpath, _, filenames in sorted(os.walk(models_dir)):
        for filename in sorted(filenames):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in MODEL_EXTENSIONS:
                continue

            filepath = os.path.join(dirpath, filename)
            size = os.path.getsize(filepath)
            reldir = os.path.relpath(dirpath, models_dir)
            if reldir == ".":
                reldir = "(root)"

            groups.setdefault(reldir, []).append((filename, size))
            total_size += size
            total_count += 1

    if not groups:
        print("No model files found.")
        return

    for subdir in sorted(groups.keys()):
        files = groups[subdir]
        dir_size = sum(s for _, s in files)
        print(f"[{subdir}] ({len(files)} file{'s' if len(files) != 1 else ''}, {format_size(dir_size)})")
        for filename, size in files:
            print(f"  {filename:55s} {format_size(size):>8s}")
        print()

    print(f"Total: {total_count} model file{'s' if total_count != 1 else ''}, {format_size(total_size)}")


if __name__ == "__main__":
    main()
