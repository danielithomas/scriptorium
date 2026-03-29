#!/usr/bin/env python3
"""Stage 5: Video Segments — Create per-slide MP4 segments with optional text overlays."""

import json
import os
import subprocess
import sys


def build_overlay_filter(overlay: dict, width: int, height: int) -> str:
    """Build FFmpeg drawtext filter from overlay config."""
    text = overlay.get("text", "").replace("'", "'\\''").replace(":", "\\:")
    font_size = overlay.get("font_size", 48)
    font_color = overlay.get("font_color", "white")
    position = overlay.get("position", "bottom-centre")

    # Position mapping
    positions = {
        "top-left":      "x=40:y=40",
        "top-centre":    "x=(w-text_w)/2:y=40",
        "top-right":     "x=w-text_w-40:y=40",
        "centre":        "x=(w-text_w)/2:y=(h-text_h)/2",
        "bottom-left":   "x=40:y=h-text_h-60",
        "bottom-centre": "x=(w-text_w)/2:y=h-text_h-60",
        "bottom-right":  "x=w-text_w-40:y=h-text_h-60",
    }

    pos = positions.get(position, positions["bottom-centre"])

    return (
        f"drawtext=text='{text}':"
        f"fontsize={font_size}:"
        f"fontcolor={font_color}:"
        f"borderw=2:bordercolor=black@0.6:"
        f"{pos}"
    )


def main():
    script_path = sys.argv[1]
    working_dir = sys.argv[2]

    with open(script_path) as f:
        script = json.load(f)

    slides_dir = os.path.join(working_dir, "slides")
    narration_dir = os.path.join(working_dir, "narration")
    segments_dir = os.path.join(working_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    resolution = script.get("resolution", "1920x1080")
    width, height = [int(x) for x in resolution.split("x")]
    default_dur = script.get("default_slide_duration", 5)
    encoder = os.environ.get("FFMPEG_ENCODER", "libx264")

    for slide in script["slides"]:
        sid = slide["id"]
        dur = slide.get("duration") or default_dur
        image = os.path.join(slides_dir, f"slide_{sid:04d}.png")
        output = os.path.join(segments_dir, f"segment_{sid:04d}.mp4")

        # Check if we have narration — if so, use its duration as minimum
        narr_file = os.path.join(narration_dir, f"narration_{sid:04d}.wav")
        has_narration = os.path.exists(narr_file)

        # Build FFmpeg command
        cmd = ["ffmpeg", "-y"]

        # Input: loop the still image
        cmd.extend(["-loop", "1", "-i", image])

        # Video filter
        vf_parts = [f"fps=30,format=yuv420p"]

        # Add text overlay if configured
        overlay = slide.get("overlay")
        if overlay and overlay.get("text"):
            vf_parts.append(build_overlay_filter(overlay, width, height))

        vf = ",".join(vf_parts)

        cmd.extend([
            "-vf", vf,
            "-c:v", encoder,
            "-t", str(dur),
            "-pix_fmt", "yuv420p",
        ])

        # Encoder-specific settings
        if encoder == "h264_nvenc":
            cmd.extend(["-preset", "p4", "-rc", "vbr", "-cq", "23"])
        else:
            cmd.extend(["-preset", "medium", "-crf", "23"])

        cmd.append(output)

        subprocess.run(cmd, check=True, capture_output=True)
        overlay_note = " +overlay" if (overlay and overlay.get("text")) else ""
        print(f"  ✓ Slide {sid}: {dur}s{overlay_note} → {os.path.basename(output)}")

    print(f"  Generated {len(script['slides'])} video segments")


if __name__ == "__main__":
    main()
