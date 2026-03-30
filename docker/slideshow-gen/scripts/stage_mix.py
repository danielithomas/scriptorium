#!/usr/bin/env python3
"""Stage 4: Audio Mixing — Combine per-slide narration with background music.

Reads durations.json from narration stage for actual slide timings.
"""

import json
import os
import subprocess
import sys


def load_durations(working_dir: str, script: dict) -> list[dict]:
    """Load slide durations from narration manifest or fall back to script."""
    manifest_path = os.path.join(working_dir, "narration", "durations.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"  Using narration-driven durations (total: {manifest['total_duration']:.1f}s)")
        return manifest["slides"]

    # Fallback: build from script
    default_dur = script.get("default_slide_duration", 5)
    return [
        {
            "slide_id": s["id"],
            "duration": s.get("duration") or default_dur,
            "narration_file": f"narration_{s['id']:04d}.wav",
        }
        for s in script["slides"]
    ]


def main():
    script_path = sys.argv[1]
    working_dir = sys.argv[2]

    volume = "0.2"
    for arg in sys.argv[3:]:
        if arg.startswith("--volume="):
            volume = arg.split("=", 1)[1]

    with open(script_path) as f:
        script = json.load(f)

    narration_dir = os.path.join(working_dir, "narration")
    music_path = os.path.join(working_dir, "music.wav")
    output_path = os.path.join(working_dir, "mixed.wav")

    no_music = not os.path.exists(music_path)

    # Load actual durations from narration stage
    slide_durations = load_durations(working_dir, script)

    # Calculate intro/outro durations for total audio timeline
    intro_dur = 0.0
    if script.get("intro"):
        intro_dur = script["intro"].get("duration", 3)

    outro_dur = 0.0
    if script.get("outro"):
        outro_dur = script["outro"].get("duration", 3)

    # Music fade-out duration (seconds before end of video)
    music_fade_out = script.get("music_fade_out", 0)

    # Crossfade overlap: each crossfade removes this many seconds from total
    crossfade_dur = script.get("crossfade_duration", 0.5) if script.get("crossfade", True) else 0
    num_content_slides = len(script.get("slides", []))
    # Number of crossfade transitions: between intro→slide1, slide1→slide2, ..., slideN→outro
    num_segments = num_content_slides + (1 if intro_dur > 0 else 0) + (1 if outro_dur > 0 else 0)
    num_transitions = max(0, num_segments - 1) if crossfade_dur > 0 else 0

    # Build narration segments with correct timing offsets
    # Narration starts after the intro (minus crossfade overlap into first slide)
    segments = []
    narr_offset = intro_dur - (crossfade_dur if intro_dur > 0 else 0)
    if narr_offset < 0:
        narr_offset = 0

    content_duration = 0.0
    for sd in slide_durations:
        sid = sd["slide_id"]
        dur = sd["duration"]
        narr_file = os.path.join(narration_dir, f"narration_{sid:04d}.wav")

        if os.path.exists(narr_file):
            segments.append({
                "file": narr_file,
                "offset": narr_offset,
                "slide_duration": dur,
            })

        narr_offset += dur
        content_duration += dur

    # Total audio duration = intro + content + outro - crossfade overlaps
    total_duration = intro_dur + content_duration + outro_dur - (num_transitions * crossfade_dur)
    if total_duration < 0:
        total_duration = content_duration  # safety fallback

    print(f"  Timeline: intro={intro_dur}s + content={content_duration:.1f}s + outro={outro_dur}s"
          f" - {num_transitions}×{crossfade_dur}s xfade = {total_duration:.1f}s total")

    if not segments and no_music:
        print("  No narration or music — skipping mix stage")
        return

    # Use FFmpeg to build the mixed audio
    filter_parts = []
    inputs = []

    # Add narration files as inputs
    for i, seg in enumerate(segments):
        inputs.extend(["-i", seg["file"]])
        delay_ms = int(seg["offset"] * 1000)
        filter_parts.append(f"[{i}]adelay={delay_ms}|{delay_ms}[narr{i}]")

    # Mix all narration streams
    if segments:
        narr_labels = "".join(f"[narr{i}]" for i in range(len(segments)))
        filter_parts.append(f"{narr_labels}amix=inputs={len(segments)}:duration=longest[narration]")
    else:
        filter_parts.append(f"anullsrc=r=24000:cl=mono,atrim=0:{total_duration}[narration]")

    # Add music and mix with narration
    if not no_music:
        music_idx = len(segments)
        inputs.extend(["-i", music_path])

        # Music plays for total_duration, with optional fade-out at the end
        music_filter = (
            f"[{music_idx}]atrim=0:{total_duration},asetpts=PTS-STARTPTS,"
            f"volume={volume}"
        )
        if music_fade_out > 0:
            fade_start = max(0, total_duration - music_fade_out)
            music_filter += f",afade=t=out:st={fade_start:.3f}:d={music_fade_out:.3f}"
            print(f"  Music fade-out: {music_fade_out}s starting at {fade_start:.1f}s")

        music_filter += "[bgmusic]"
        filter_parts.append(music_filter)

        filter_parts.append(
            "[narration][bgmusic]amix=inputs=2:duration=longest:dropout_transition=2[out]"
        )
        output_label = "[out]"
    else:
        output_label = "[narration]"

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", output_label,
        "-ar", "24000",
        "-ac", "1",
        output_path,
    ]

    print(f"  Mixing {len(segments)} narration tracks" +
          (f" + background music at {float(volume)*100:.0f}% volume" if not no_music else ""))

    subprocess.run(cmd, check=True, capture_output=True)
    print(f"  ✓ Mixed audio: {total_duration:.1f}s → {os.path.basename(output_path)}")


if __name__ == "__main__":
    main()
