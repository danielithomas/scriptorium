#!/usr/bin/env python3
"""Stage 1: Image Preparation — Normalise input images to consistent resolution."""

import json
import os
import subprocess
import sys

def main():
    script_path = sys.argv[1]
    output_dir = sys.argv[2]

    with open(script_path) as f:
        script = json.load(f)

    resolution = script.get("resolution", "1920x1080")
    width, height = resolution.split("x")
    input_dir = os.path.join(os.path.dirname(script_path), "images")

    os.makedirs(output_dir, exist_ok=True)

    for slide in script["slides"]:
        sid = slide["id"]
        src = os.path.join(input_dir, slide["image"])
        dst = os.path.join(output_dir, f"slide_{sid:04d}.png")

        if not os.path.exists(src):
            print(f"  WARNING: Image not found: {src}, generating black frame")
            subprocess.run([
                "magick", "convert",
                "-size", f"{width}x{height}",
                "xc:black", dst
            ], check=True)
        else:
            subprocess.run([
                "magick", "convert", src,
                "-resize", f"{width}x{height}",
                "-background", "black",
                "-gravity", "center",
                "-extent", f"{width}x{height}",
                dst
            ], check=True)
            print(f"  ✓ Slide {sid}: {slide['image']} → {os.path.basename(dst)}")

    print(f"  Prepared {len(script['slides'])} slides at {resolution}")


if __name__ == "__main__":
    main()
