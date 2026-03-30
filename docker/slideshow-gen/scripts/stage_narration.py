#!/usr/bin/env python3
"""Stage 1: Narration Synthesis — Generate per-slide WAV audio via TTS.

Routes to Chatterbox TTS for English, IndicF5 for Hindi/Punjabi/Gujarati.

Outputs:
  - narration_XXXX.wav per slide (in output_dir)
  - durations.json — actual narration durations + computed slide timings

Duration logic:
  - If narration audio is longer than the script duration, use narration + padding
  - If narration is shorter or absent, use script duration (or default_slide_duration)
  - Video slides have fixed duration; if narration overflows, excess carries to next slide
"""

import json
import os
import sys
import time

import numpy as np
import soundfile as sf

PADDING_SECONDS = 0.5  # Breathing room after narration ends
MIN_SLIDE_DURATION = 6  # Minimum seconds for any slide (can be overridden by script)


def get_device():
    """Get the configured inference device."""
    return os.environ.get("TTS_DEVICE", os.environ.get("DEVICE", "cpu"))


def load_chatterbox(device: str):
    """Load Chatterbox TTS model."""
    print("  Loading Chatterbox TTS (English)...")
    from chatterbox.tts import ChatterboxTTS
    model = ChatterboxTTS.from_pretrained(device=device)
    return model


def load_indicf5(device: str):
    """Load IndicF5 model for Hindi/Punjabi."""
    print("  Loading IndicF5 (Hindi/Punjabi)...")
    from transformers import AutoModel
    import torch

    model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)
    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    return model


def synthesise_english(model, text: str, voice_ref: str | None, output_path: str):
    """Generate English narration with Chatterbox."""
    if voice_ref:
        audio = model.generate(text, audio_prompt_path=voice_ref)
    else:
        audio = model.generate(text)

    audio_np = audio.squeeze().cpu().numpy()
    sf.write(output_path, audio_np, 24000)


def synthesise_indic(model, text: str, language: str, voice_ref: str | None,
                     ref_text: str | None, output_path: str):
    """Generate Hindi/Punjabi/Gujarati narration with IndicF5."""
    kwargs = {"text": text}

    if voice_ref and ref_text:
        kwargs["ref_audio_path"] = voice_ref
        kwargs["ref_text"] = ref_text
    elif voice_ref:
        kwargs["ref_audio_path"] = voice_ref
        kwargs["ref_text"] = ""

    audio = model(**kwargs)

    if hasattr(audio, "dtype") and audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    sf.write(output_path, np.array(audio, dtype=np.float32), 24000)


def get_wav_duration(path: str) -> float:
    """Get duration of a WAV file in seconds."""
    info = sf.info(path)
    return info.duration


def main():
    script_path = sys.argv[1]
    output_dir = sys.argv[2]

    voice_ref = None
    for arg in sys.argv[3:]:
        if arg.startswith("--voice-ref="):
            voice_ref = arg.split("=", 1)[1]

    with open(script_path) as f:
        script = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    default_lang = script.get("default_language", "en")
    default_dur = script.get("default_slide_duration", MIN_SLIDE_DURATION)

    # Determine which engines are needed
    languages_needed = set()
    for slide in script["slides"]:
        if slide.get("narration"):
            lang = slide.get("language", default_lang)
            languages_needed.add(lang)

    # Load models on demand
    chatterbox_model = None
    indicf5_model = None

    if "en" in languages_needed:
        chatterbox_model = load_chatterbox(device)
    if languages_needed & {"hi", "pa", "gu"}:
        indicf5_model = load_indicf5(device)

    # ── Phase 1: Generate all narration audio ──────────────────
    narration_durations = {}  # sid -> float seconds

    for slide in script["slides"]:
        sid = slide["id"]
        narration = slide.get("narration", "")
        lang = slide.get("language", default_lang)

        if not narration:
            print(f"  ○ Slide {sid}: No narration")
            continue

        output_path = os.path.join(output_dir, f"narration_{sid:04d}.wav")
        start = time.time()

        if lang == "en":
            synthesise_english(chatterbox_model, narration, voice_ref, output_path)
        elif lang in ("hi", "pa", "gu"):
            ref_text = slide.get("ref_text")
            synthesise_indic(indicf5_model, narration, lang, voice_ref, ref_text, output_path)
        else:
            print(f"  WARNING: Unsupported language '{lang}' for slide {sid}, skipping")
            continue

        elapsed = time.time() - start
        narr_dur = get_wav_duration(output_path)
        narration_durations[sid] = narr_dur
        print(f"  ✓ Slide {sid} [{lang}]: {elapsed:.1f}s gen, {narr_dur:.1f}s audio → {os.path.basename(output_path)}")

    # ── Phase 2: Compute actual slide timings ──────────────────
    # Rules:
    #   - Non-video slides: max(script_duration, narration_duration + padding, default)
    #   - Video slides: fixed duration from video file; overflow narration carries forward
    #   - Overflow: if narration for a video slide exceeds its fixed duration,
    #     the excess is tracked and delays the next slide's narration start

    slide_timings = []
    narration_overflow = 0.0  # Seconds of narration overflow from previous video slide

    for slide in script["slides"]:
        sid = slide["id"]
        is_video = slide.get("type") == "video"
        script_dur = slide.get("duration") or default_dur
        narr_dur = narration_durations.get(sid, 0.0)

        if is_video:
            # Video slides have fixed duration — can't be stretched
            actual_dur = script_dur
            if narr_dur > script_dur:
                # Narration overflows into next slide
                narration_overflow = narr_dur - script_dur
                print(f"  ⚠ Slide {sid} (video): narration ({narr_dur:.1f}s) overflows by {narration_overflow:.1f}s")
            else:
                narration_overflow = 0.0
        else:
            # Image slides: stretch to fit narration
            # Account for any overflow from previous video slide
            effective_narr = narr_dur + narration_overflow
            actual_dur = max(script_dur, effective_narr + PADDING_SECONDS if effective_narr > 0 else script_dur)
            narration_overflow = 0.0  # Absorbed

        timing = {
            "slide_id": sid,
            "duration": round(actual_dur, 2),
            "narration_duration": round(narr_dur, 2),
            "is_video": is_video,
            "narration_file": f"narration_{sid:04d}.wav" if narr_dur > 0 else None,
        }
        slide_timings.append(timing)

    total_duration = sum(t["duration"] for t in slide_timings)

    # Add intro/outro durations
    intro_dur = script.get("intro", {}).get("duration", 0)
    outro_dur = script.get("outro", {}).get("duration", 0)
    total_with_bookends = total_duration + intro_dur + outro_dur

    # Write durations manifest
    manifest = {
        "total_duration": round(total_with_bookends, 2),
        "content_duration": round(total_duration, 2),
        "intro_duration": intro_dur,
        "outro_duration": outro_dur,
        "default_slide_duration": default_dur,
        "padding_seconds": PADDING_SECONDS,
        "slides": slide_timings,
    }

    manifest_path = os.path.join(output_dir, "durations.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Narration complete: {len(narration_durations)} slides voiced")
    print(f"  Total video duration: {total_duration:.1f}s")
    print(f"  Durations manifest: {manifest_path}")


if __name__ == "__main__":
    main()
