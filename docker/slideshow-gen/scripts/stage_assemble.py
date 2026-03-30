#!/usr/bin/env python3
"""Stage 6: Final Assembly — Concatenate video segments and merge with mixed audio."""

import json
import os
import subprocess
import sys
import tempfile


def generate_srt(script: dict, output_path: str, working_dir: str):
    """Generate SRT subtitle file from slide narrations using actual durations."""
    # Try to load narration-driven durations
    durations = {}
    manifest_path = os.path.join(working_dir, "narration", "durations.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        durations = {s["slide_id"]: s["duration"] for s in manifest["slides"]}

    default_dur = script.get("default_slide_duration", 5)
    offset = 0.0
    subs = []

    for i, slide in enumerate(script["slides"], 1):
        dur = durations.get(slide["id"], slide.get("duration") or default_dur)
        narration = slide.get("narration", "")

        if narration:
            start_h = int(offset // 3600)
            start_m = int((offset % 3600) // 60)
            start_s = offset % 60

            end = offset + dur
            end_h = int(end // 3600)
            end_m = int((end % 3600) // 60)
            end_s = end % 60

            subs.append(
                f"{i}\n"
                f"{start_h:02d}:{start_m:02d}:{start_s:06.3f} --> "
                f"{end_h:02d}:{end_m:02d}:{end_s:06.3f}\n"
                f"{narration}\n"
            )

        offset += dur

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(subs))


def main():
    script_path = sys.argv[1]
    working_dir = sys.argv[2]
    output_path = sys.argv[3]

    subtitles = "--subtitles" in sys.argv

    with open(script_path) as f:
        script = json.load(f)

    segments_dir = os.path.join(working_dir, "segments")
    mixed_audio = os.path.join(working_dir, "mixed.wav")
    encoder = os.environ.get("FFMPEG_ENCODER", "libx264")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Build concat list
    segment_files = sorted([
        os.path.join(segments_dir, f)
        for f in os.listdir(segments_dir)
        if f.endswith(".mp4")
    ])

    if not segment_files:
        print("  ERROR: No video segments found")
        sys.exit(1)

    # Write concat file
    concat_file = os.path.join(working_dir, "concat.txt")
    with open(concat_file, "w") as f:
        for seg in segment_files:
            f.write(f"file '{seg}'\n")

    # Assemble final video
    cmd = ["ffmpeg", "-y"]

    # Concat video segments
    cmd.extend(["-f", "concat", "-safe", "0", "-i", concat_file])

    # Add mixed audio if available
    has_audio = os.path.exists(mixed_audio)
    if has_audio:
        cmd.extend(["-i", mixed_audio])

    # Video: copy (already encoded)
    cmd.extend(["-c:v", "copy"])

    # Audio
    if has_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-ar", "44100"])
        cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])
    else:
        cmd.extend(["-an"])

    cmd.extend(["-movflags", "+faststart"])
    cmd.append(output_path)

    subprocess.run(cmd, check=True, capture_output=True)

    # Get file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✓ Final video: {output_path} ({size_mb:.1f} MB)")

    # Generate subtitles if requested
    if subtitles:
        srt_path = output_path.rsplit(".", 1)[0] + ".srt"
        generate_srt(script, srt_path, working_dir)
        print(f"  ✓ Subtitles: {srt_path}")


if __name__ == "__main__":
    main()
