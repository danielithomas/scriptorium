# Slideshow Video Generator

CLI-driven pipeline for generating slideshow videos with AI narration (text-to-speech) and AI background music (text-to-music). Runs fully offline — no cloud dependencies.

## Features

- **Multilingual TTS**: English (Chatterbox), Hindi & Punjabi (IndicF5)
- **AI Music**: Background music generation via Meta MusicGen
- **Text Overlays**: Configurable per-slide text with positioning
- **Subtitles**: Optional SRT generation
- **Voice Cloning**: Reference audio support for consistent speaker voice
- **Dual Containers**: CPU-only and NVIDIA GPU variants

## Quick Start

### 1. Prepare Input

```bash
mkdir -p input/images output models

# Add your slide images
cp ~/slides/*.jpg input/images/

# Create script.json (see below)
# Optionally create input/music.txt with a music prompt
```

### 2. Build & Run

**CPU variant:**
```bash
docker compose -f compose-cpu.yaml build
docker compose -f compose-cpu.yaml up
```

**GPU variant (requires nvidia-container-toolkit):**
```bash
docker compose -f compose-gpu.yaml build
docker compose -f compose-gpu.yaml up
```

Output: `./output/final.mp4`

## Script Format (`script.json`)

```json
{
  "title": "My Presentation",
  "resolution": "1920x1080",
  "default_slide_duration": 6,
  "default_language": "en",
  "slides": [
    {
      "id": 1,
      "image": "slide01.jpg",
      "language": "en",
      "narration": "Welcome to our presentation.",
      "duration": null,
      "overlay": {
        "text": "Introduction",
        "position": "bottom-centre",
        "font_size": 48,
        "font_color": "white"
      }
    },
    {
      "id": 2,
      "image": "slide02.jpg",
      "language": "hi",
      "narration": "यह प्रस्तुति एआई टूलिंग के बारे में है।",
      "ref_text": "यह संदर्भ वाक्य है।"
    }
  ]
}
```

### Language Codes

| Code | Engine | Notes |
|------|--------|-------|
| `en` | Chatterbox TTS | English, voice cloning via `--voice-ref` |
| `hi` | IndicF5 | Hindi, optional `ref_text` per slide |
| `pa` | IndicF5 | Punjabi, optional `ref_text` per slide |
| `gu` | IndicF5 | Gujarati, optional `ref_text` per slide |

### Overlay Positions

`top-left`, `top-centre`, `top-right`, `centre`, `bottom-left`, `bottom-centre`, `bottom-right`

## CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Path to script.json | `/input/script.json` |
| `--output` | Output MP4 path | `/output/final.mp4` |
| `--music-prompt` | Override music prompt | From `music.txt` |
| `--music-duration` | Music length (seconds) | Auto (match video) |
| `--music-volume` | Music vs narration (0–1) | `0.2` |
| `--no-music` | Skip music generation | `false` |
| `--no-narration` | Skip TTS narration | `false` |
| `--voice-ref` | Reference WAV for voice cloning | None |
| `--subtitles` | Generate .srt file | `false` |
| `--keep-working` | Keep intermediate files | `false` |

### Example: Custom Run

```bash
docker compose -f compose-gpu.yaml run slideshow-gpu \
  --input=/input/script.json \
  --output=/output/final.mp4 \
  --voice-ref=/input/voice_ref.wav \
  --subtitles \
  --keep-working
```

## Environment Variables

| Variable | CPU Default | GPU Default |
|----------|-------------|-------------|
| `DEVICE` | `cpu` | `cuda` |
| `TTS_DEVICE` | `cpu` | `cuda` |
| `MUSICGEN_MODEL` | `facebook/musicgen-small` | `facebook/musicgen-large` |
| `FFMPEG_ENCODER` | `libx264` | `h264_nvenc` |
| `NUM_THREADS` | `12` | `24` |
| `MUSIC_VOLUME` | `0.2` | `0.2` |
| `HF_TOKEN` | (optional) | (optional) |

## Pipeline Stages

1. **Image Preparation** — Normalise to target resolution via ImageMagick
2. **Narration Synthesis** — Per-slide TTS (Chatterbox / IndicF5 by language)
3. **Music Generation** — Background music via MusicGen (30s clips, auto-stitched)
4. **Audio Mixing** — Combine narration + music with FFmpeg
5. **Text Overlays** — Render per-slide text overlays
6. **Final Assembly** — Concatenate segments + mixed audio → MP4

## Model Cache

Models are downloaded on first run to the `/models` volume. Approximate sizes:

| Model | Size |
|-------|------|
| MusicGen-small | ~1.5 GB |
| MusicGen-large | ~8 GB |
| Chatterbox TTS | ~1–2 GB |
| IndicF5 | ~3–5 GB |

## Performance

| Metric | CPU | GPU (RTX 5080) |
|--------|-----|----------------|
| TTS per slide | 10–30s | 1–3s |
| Music (30s clip) | 1–5 min | 5–10s |
| Total (10 slides) | ~10–15 min | ~1–2 min |

## Host Prerequisites (GPU)

```bash
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Requires NVIDIA driver ≥ 570 and CUDA ≥ 12.4.
