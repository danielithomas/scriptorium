#!/usr/bin/env python3
"""Stage 6: Final Assembly — Assemble video segments with crossfades and merge audio.

Supports optional crossfade transitions between segments using FFmpeg xfade filter.
"""

import json
import os
import subprocess
import sys


def generate_srt(script: dict, output_path: str, working_dir: str):
    """Generate SRT subtitle file from slide narrations using actual durations."""
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


def get_segment_duration(path: str) -> float:
    """Get duration of a video segment using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def assemble_with_xfade(segment_files: list[str], fade_duration: float,
                        output_path: str, encoder: str):
    """Assemble segments with xfade crossfade transitions."""
    n = len(segment_files)

    if n == 1:
        # Single segment — just copy
        subprocess.run(
            ["ffmpeg", "-y", "-i", segment_files[0], "-c", "copy", output_path],
            check=True, capture_output=True,
        )
        return

    # Get durations for offset calculation
    durations = [get_segment_duration(f) for f in segment_files]

    # Build xfade filter chain
    # Each xfade needs: offset = cumulative duration - (num_transitions_so_far * fade_duration) - fade_duration
    inputs = []
    for f in segment_files:
        inputs.extend(["-i", f])

    filter_parts = []
    cumulative = durations[0]

    for i in range(1, n):
        offset = cumulative - fade_duration
        if offset < 0:
            offset = 0

        if i == 1:
            src = "[0:v][1:v]"
        else:
            src = f"[xf{i-1}][{i}:v]"

        if i == n - 1:
            dst = "[vout]"
        else:
            dst = f"[xf{i}]"

        filter_parts.append(
            f"{src}xfade=transition=fade:duration={fade_duration}:offset={offset:.3f}{dst}"
        )

        cumulative += durations[i] - fade_duration

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-c:v", encoder,
        "-pix_fmt", "yuv420p",
    ]

    if encoder == "h264_nvenc":
        cmd.extend(["-preset", "p4", "-rc", "vbr", "-cq", "23"])
    else:
        cmd.extend(["-preset", "medium", "-crf", "23"])

    cmd.append(output_path)
    subprocess.run(cmd, check=True, capture_output=True)


def assemble_simple(segment_files: list[str], output_path: str):
    """Assemble segments with simple concatenation (no transitions)."""
    concat_file = output_path + ".concat.txt"
    with open(concat_file, "w") as f:
        for seg in segment_files:
            f.write(f"file '{seg}'\n")

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
         "-c:v", "copy", output_path],
        check=True, capture_output=True,
    )
    os.remove(concat_file)


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

    # Gather all segment files in order
    segment_files = sorted([
        os.path.join(segments_dir, f)
        for f in os.listdir(segments_dir)
        if f.endswith(".mp4")
    ])

    if not segment_files:
        print("  ERROR: No video segments found")
        sys.exit(1)

    # Check for crossfade config
    fade_duration = script.get("crossfade_duration", 0.5)
    use_fades = script.get("crossfade", True)  # Default: enabled

    # Assemble video (to temp file if we need to add audio)
    has_audio = os.path.exists(mixed_audio)
    video_only = output_path + ".video.mp4" if has_audio else output_path

    if use_fades and len(segment_files) > 1:
        print(f"  Assembling {len(segment_files)} segments with {fade_duration}s crossfades")
        assemble_with_xfade(segment_files, fade_duration, video_only, encoder)
    else:
        print(f"  Assembling {len(segment_files)} segments (no transitions)")
        assemble_simple(segment_files, video_only)

    # Merge with audio
    if has_audio:
        print(f"  Merging audio track...")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_only,
            "-i", mixed_audio,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            "-movflags", "+faststart",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(video_only)
    else:
        # No audio — add faststart flag
        if video_only != output_path:
            os.rename(video_only, output_path)

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
