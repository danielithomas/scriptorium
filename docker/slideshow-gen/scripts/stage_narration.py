#!/usr/bin/env python3
"""Stage 2: Narration Synthesis — Generate per-slide WAV audio via TTS.

Routes to Chatterbox TTS for English, IndicF5 for Hindi/Punjabi.
"""

import json
import os
import sys
import time

import numpy as np
import soundfile as sf


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

    # Chatterbox returns a tensor; convert to numpy
    audio_np = audio.squeeze().cpu().numpy()
    sf.write(output_path, audio_np, 24000)


def synthesise_indic(model, text: str, language: str, voice_ref: str | None,
                     ref_text: str | None, output_path: str):
    """Generate Hindi/Punjabi narration with IndicF5."""
    kwargs = {"text": text}

    if voice_ref and ref_text:
        kwargs["ref_audio_path"] = voice_ref
        kwargs["ref_text"] = ref_text
    elif voice_ref:
        # Use reference audio without transcript (model will handle it)
        kwargs["ref_audio_path"] = voice_ref
        kwargs["ref_text"] = ""

    audio = model(**kwargs)

    # Normalise if int16
    if hasattr(audio, "dtype") and audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    sf.write(output_path, np.array(audio, dtype=np.float32), 24000)


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
    if languages_needed & {"hi", "pa"}:
        indicf5_model = load_indicf5(device)

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
        elif lang in ("hi", "pa"):
            ref_text = slide.get("ref_text")
            synthesise_indic(indicf5_model, narration, lang, voice_ref, ref_text, output_path)
        else:
            print(f"  WARNING: Unsupported language '{lang}' for slide {sid}, skipping")
            continue

        elapsed = time.time() - start
        print(f"  ✓ Slide {sid} [{lang}]: {elapsed:.1f}s → {os.path.basename(output_path)}")

    print(f"  Narration complete for {len(script['slides'])} slides")


if __name__ == "__main__":
    main()
