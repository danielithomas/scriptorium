# Slideshow Video Generator

Persistent API service for generating slideshow videos with AI narration (text-to-speech) and AI background music (text-to-music). Runs fully offline — no cloud dependencies.

## Features

- **REST API** — Submit jobs via HTTP, poll progress, download results
- **Job Queue** — Sequential GPU processing with progress tracking per stage
- **Multilingual TTS** — English (Chatterbox), Hindi, Punjabi & Gujarati (IndicF5)
- **AI Music** — Background music generation via Meta MusicGen
- **Text Overlays** — Configurable per-slide text with positioning
- **Subtitles** — Optional SRT generation
- **Voice Cloning** — Reference audio support for consistent speaker voice
- **Dual Containers** — CPU-only and NVIDIA GPU variants
- **CLI Mode** — Still supports one-shot CLI execution via entrypoint.sh

## Quick Start

### 1. Build & Run (API Server)

**GPU variant (recommended for production):**
```bash
docker compose -f compose-gpu.yaml build
docker compose -f compose-gpu.yaml up -d
```

**CPU variant:**
```bash
docker compose -f compose-cpu.yaml build
docker compose -f compose-cpu.yaml up -d
```

The API is available at `http://localhost:8080`.

### 2. Submit a Job

```bash
# Simple job with script + images
curl -X POST http://localhost:8080/generate \
  -F "script=@input/script.json" \
  -F "images=@input/images/slide01.jpg" \
  -F "images=@input/images/slide02.jpg" \
  -F "images=@input/images/slide03.jpg"

# With voice reference and music prompt
curl -X POST http://localhost:8080/generate \
  -F "script=@input/script.json" \
  -F "images=@input/images/slide01.jpg" \
  -F "images=@input/images/slide02.jpg" \
  -F "voice_ref=@input/voice_ref.wav" \
  -F "music_prompt=gentle acoustic guitar, warm and hopeful"

# Skip music, add subtitles
curl -X POST http://localhost:8080/generate \
  -F "script=@input/script.json" \
  -F "images=@input/images/slide01.jpg" \
  -F "no_music=true" \
  -F "subtitles=true"
```

Response:
```json
{"job_id": "a1b2c3d4e5f6", "status": "queued"}
```

### 3. Check Progress

```bash
curl http://localhost:8080/jobs/a1b2c3d4e5f6
```

Response:
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "running",
  "progress": {"stage": 3, "total": 6, "name": "Music Generation"},
  "created_at": "2026-03-30T00:00:00+00:00",
  "started_at": "2026-03-30T00:00:01+00:00",
  "completed_at": null,
  "error": null,
  "options": {
    "no_music": false,
    "no_narration": false,
    "subtitles": false,
    "has_voice_ref": true,
    "slide_count": 10,
    "title": "My Presentation"
  }
}
```

### 4. Download Result

```bash
curl -o slideshow.mp4 http://localhost:8080/jobs/a1b2c3d4e5f6/output
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health + GPU status |
| `POST` | `/generate` | Submit a new slideshow job |
| `GET` | `/jobs` | List recent jobs (default: 20) |
| `GET` | `/jobs/{id}` | Get job status and progress |
| `GET` | `/jobs/{id}/output` | Download completed MP4 |
| `DELETE` | `/jobs/{id}` | Delete completed/failed job |

### POST /generate — Multipart Form Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `script` | file | ✅ | `script.json` — slide definitions |
| `images` | file[] | ✅ | Slide image files (one per slide) |
| `voice_ref` | file | ❌ | Voice reference WAV for cloning |
| `music_prompt` | string | ❌ | Text prompt for music generation |
| `no_music` | bool | ❌ | Skip music generation (default: false) |
| `no_narration` | bool | ❌ | Skip TTS narration (default: false) |
| `subtitles` | bool | ❌ | Generate SRT subtitles (default: false) |

### Job Status Values

| Status | Meaning |
|--------|---------|
| `queued` | Waiting in queue |
| `running` | Pipeline executing (check `progress` for stage) |
| `completed` | Done — output available at `/jobs/{id}/output` |
| `failed` | Error occurred — check `error` field |

## CLI Mode (Legacy)

The one-shot CLI still works for manual/scripted runs:

```bash
# Override the default CMD to use entrypoint.sh
docker compose -f compose-gpu.yaml run --rm \
  --entrypoint /app/entrypoint.sh \
  slideshow-gpu \
  --input=/input/script.json \
  --output=/output/final.mp4 \
  --voice-ref=/input/voice_ref.wav \
  --subtitles
```

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
| `en` | Chatterbox TTS | English, voice cloning via `voice_ref` |
| `hi` | IndicF5 | Hindi, optional `ref_text` per slide |
| `pa` | IndicF5 | Punjabi, optional `ref_text` per slide |
| `gu` | IndicF5 | Gujarati, optional `ref_text` per slide |

### Overlay Positions

`top-left`, `top-centre`, `top-right`, `centre`, `bottom-left`, `bottom-centre`, `bottom-right`

## Environment Variables

| Variable | CPU Default | GPU Default |
|----------|-------------|-------------|
| `API_PORT` | `8080` | `8080` |
| `MAX_STORED_JOBS` | `20` | `20` |
| `DEVICE` | `cpu` | `cuda` |
| `TTS_DEVICE` | `cpu` | `cuda` |
| `MUSICGEN_MODEL` | `musicgen-small` | `musicgen-large` |
| `FFMPEG_ENCODER` | `libx264` | `h264_nvenc` |
| `NUM_THREADS` | `12` | `24` |
| `MUSIC_VOLUME` | `0.2` | `0.2` |
| `HF_TOKEN` | (optional) | (optional) |

## Pipeline Stages

1. **Image Preparation** — Normalise to target resolution via ImageMagick
2. **Narration Synthesis** — Per-slide TTS (Chatterbox / IndicF5 by language)
3. **Music Generation** — Background music via MusicGen (30s clips, auto-stitched)
4. **Audio Mixing** — Combine narration + music with FFmpeg
5. **Video Segments + Overlays** — Render per-slide text overlays
6. **Final Assembly** — Concatenate segments + mixed audio → MP4

## Data & Storage

Jobs are stored in `/data/jobs/{job_id}/`:
```
/data/jobs/a1b2c3d4e5f6/
├── input/
│   ├── script.json
│   ├── images/
│   ├── voice_ref.wav (optional)
│   └── music.txt (optional)
├── output/
│   └── final.mp4
└── status.json
```

Old completed/failed jobs are automatically pruned beyond `MAX_STORED_JOBS` (default: 20).
Jobs interrupted by a restart are automatically re-queued on startup.

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
