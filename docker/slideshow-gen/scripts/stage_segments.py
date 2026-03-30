#!/usr/bin/env python3
"""Stage 5: Video Segments — Create per-slide MP4 segments with optional text overlays.

Reads durations.json from narration stage for actual slide timings.
"""

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


def load_durations(working_dir: str, script: dict) -> dict:
    """Load slide durations from narration manifest. Returns {slide_id: duration}."""
    manifest_path = os.path.join(working_dir, "narration", "durations.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        return {s["slide_id"]: s["duration"] for s in manifest["slides"]}

    # Fallback
    default_dur = script.get("default_slide_duration", 5)
    return {s["id"]: s.get("duration") or default_dur for s in script["slides"]}


def main():
    script_path = sys.argv[1]
    working_dir = sys.argv[2]

    with open(script_path) as f:
        script = json.load(f)

    slides_dir = os.path.join(working_dir, "slides")
    segments_dir = os.path.join(working_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    resolution = script.get("resolution", "1920x1080")
    width, height = [int(x) for x in resolution.split("x")]
    encoder = os.environ.get("FFMPEG_ENCODER", "libx264")

    # Load narration-driven durations
    durations = load_durations(working_dir, script)

    for slide in script["slides"]:
        sid = slide["id"]
        is_video = slide.get("type") == "video"
        dur = durations.get(sid, script.get("default_slide_duration", 6))
        output = os.path.join(segments_dir, f"segment_{sid:04d}.mp4")

        if is_video:
            # Video slides: use original video file, fixed duration
            video_file = os.path.join(os.path.dirname(script_path), "videos", slide.get("video", ""))
            if os.path.exists(video_file):
                # Re-encode to match resolution and codec
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_file,
                    "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                           f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps=30,format=yuv420p",
                    "-c:v", encoder, "-an",
                    "-pix_fmt", "yuv420p",
                ]
                if encoder == "h264_nvenc":
                    cmd.extend(["-preset", "p4", "-rc", "vbr", "-cq", "23"])
                else:
                    cmd.extend(["-preset", "medium", "-crf", "23"])
                cmd.append(output)
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"  ✓ Slide {sid} (video): {dur}s → {os.path.basename(output)}")
            else:
                print(f"  WARNING: Video file not found for slide {sid}: {video_file}")
            continue

        # Image slides
        image = os.path.join(slides_dir, f"slide_{sid:04d}.png")

        cmd = ["ffmpeg", "-y"]
        cmd.extend(["-loop", "1", "-i", image])

        vf_parts = ["fps=30,format=yuv420p"]

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
