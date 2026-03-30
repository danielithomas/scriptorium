#!/usr/bin/env python3
"""Stage 3: Music Generation — Generate background music from a text prompt.

Uses Meta MusicGen via AudioCraft. Stitches multiple 30s clips for longer durations.
"""

import json
import os
import sys
import time

import numpy as np
import soundfile as sf


def get_device():
    return os.environ.get("DEVICE", "cpu")


def get_total_duration(script: dict, script_path: str) -> float:
    """Get total video duration from durations.json manifest (narration-driven).
    Falls back to script durations if manifest not found."""
    # Check for durations manifest from narration stage
    narration_dir = os.path.join(os.path.dirname(script_path), "..", "narration")
    manifest_path = os.path.join(narration_dir, "durations.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        total = manifest.get("total_duration", 0)
        print(f"  Using narration-driven duration: {total:.1f}s (from durations.json)")
        return total

    # Fallback to script durations
    default_dur = script.get("default_slide_duration", 5)
    total = 0.0
    for slide in script["slides"]:
        dur = slide.get("duration") or default_dur
        total += dur
    return total


def main():
    script_path = sys.argv[1]
    output_path = sys.argv[2]

    prompt = None
    target_duration = None

    for arg in sys.argv[3:]:
        if arg.startswith("--prompt="):
            prompt = arg.split("=", 1)[1]
        elif arg.startswith("--duration="):
            target_duration = float(arg.split("=", 1)[1])

    with open(script_path) as f:
        script = json.load(f)

    # Determine music prompt
    if not prompt:
        music_txt = os.path.join(os.path.dirname(script_path), "music.txt")
        if os.path.exists(music_txt):
            with open(music_txt) as f:
                prompt = f.read().strip()
        else:
            prompt = "calm ambient background music, piano and soft strings, professional, 80bpm"
            print(f"  Using default music prompt")

    # Determine target duration
    if not target_duration:
        target_duration = get_total_duration(script, script_path)

    print(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"  Target duration: {target_duration:.1f}s")

    device = get_device()
    model_name = os.environ.get("MUSICGEN_MODEL", "facebook/musicgen-small")

    print(f"  Loading MusicGen ({model_name}) on {device}...")
    from audiocraft.models import MusicGen

    model = MusicGen.get_pretrained(model_name, device=device)

    # MusicGen max is ~30s per call; stitch if needed
    MAX_CLIP = 30
    clips = []
    remaining = target_duration

    while remaining > 0:
        clip_duration = min(remaining, MAX_CLIP)
        model.set_generation_params(duration=clip_duration)

        start = time.time()
        wav = model.generate([prompt])
        elapsed = time.time() - start

        # wav shape: (batch, channels, samples) at 32kHz
        audio = wav[0].cpu().numpy()
        clips.append(audio)
        remaining -= clip_duration
        print(f"  ✓ Generated {clip_duration:.0f}s clip in {elapsed:.1f}s")

    # Concatenate clips with crossfade
    if len(clips) == 1:
        combined = clips[0]
    else:
        # Simple crossfade of 0.5s between clips
        sr = 32000
        fade_samples = int(0.5 * sr)
        combined = clips[0]

        for clip in clips[1:]:
            if combined.shape[-1] >= fade_samples and clip.shape[-1] >= fade_samples:
                # Linear crossfade
                fade_out = np.linspace(1, 0, fade_samples)
                fade_in = np.linspace(0, 1, fade_samples)

                combined[..., -fade_samples:] *= fade_out
                clip[..., :fade_samples] *= fade_in
                combined[..., -fade_samples:] += clip[..., :fade_samples]
                combined = np.concatenate([combined, clip[..., fade_samples:]], axis=-1)
            else:
                combined = np.concatenate([combined, clip], axis=-1)

    # Write output — squeeze to mono if needed
    audio_out = combined.squeeze()
    if audio_out.ndim > 1:
        audio_out = audio_out[0]  # Take first channel

    sf.write(output_path, audio_out, 32000)
    final_duration = len(audio_out) / 32000
    print(f"  Music generated: {final_duration:.1f}s → {os.path.basename(output_path)}")


if __name__ == "__main__":
    main()
