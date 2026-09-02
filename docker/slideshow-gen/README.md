# Slideshow Video Generator

Persistent API service for generating slideshow videos with AI narration (text-to-speech) and AI background music (text-to-music). Runs fully offline — no cloud dependencies.

## Features

- **REST API** — Submit jobs via HTTP, poll progress, download results
- **Job Queue** — Sequential GPU processing with progress tracking per stage
- **Multilingual TTS** — English (Chatterbox), Hindi, Punjabi & Gujarati (IndicF5)
- **AI Music** — Background music generation via Meta MusicGen
- **Text Overlays** — Configurable per-slide text with positioning
- **Branded Intro/Outro** — Optional logo segments on a solid background
- **Crossfade Transitions** — FFmpeg `xfade` between segments, with a timeline-aware music fade-out
- **Subtitles** — Optional SRT generation
- **Voice Cloning** — Reference audio support for consistent speaker voice
- **Dual Containers** — CPU-only and NVIDIA GPU variants
- **CLI Mode** — Still supports one-shot CLI execution via entrypoint.sh

## Quick Start

### 1. Build & Run (API Server)

```bash
cp .env.example .env
# Edit .env — set DATA_PATH, MODELS_PATH, and API_PORT for your host
```

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

The API is available at `http://localhost:8189` (the container listens on 8080 internally; `API_PORT` sets the host port).

### 2. Submit a Job

```bash
# Simple job with script + images
curl -X POST http://localhost:8189/generate \
  -F "script=@input/script.json" \
  -F "images=@input/images/slide01.jpg" \
  -F "images=@input/images/slide02.jpg" \
  -F "images=@input/images/slide03.jpg"

# With voice reference and music prompt
curl -X POST http://localhost:8189/generate \
  -F "script=@input/script.json" \
  -F "images=@input/images/slide01.jpg" \
  -F "images=@input/images/slide02.jpg" \
  -F "voice_ref=@input/voice_ref.wav" \
  -F "music_prompt=gentle acoustic guitar, warm and hopeful"

# Skip music, add subtitles
curl -X POST http://localhost:8189/generate \
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
curl http://localhost:8189/jobs/a1b2c3d4e5f6
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
curl -o slideshow.mp4 http://localhost:8189/jobs/a1b2c3d4e5f6/output
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
  "crossfade": true,
  "crossfade_duration": 0.5,
  "music_fade_out": 3,
  "intro": {
    "image": "logo.png",
    "duration": 3,
    "background": "white"
  },
  "outro": {
    "image": "logo.png",
    "duration": 3,
    "background": "white"
  },
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

### Top-Level Fields

| Field | Default | Description |
|-------|---------|-------------|
| `title` | — | Job title, echoed back in job status |
| `resolution` | `1920x1080` | Output resolution |
| `default_slide_duration` | `6` | Fallback duration when narration is absent or shorter |
| `default_language` | `en` | Language used for slides that don't set one |
| `crossfade` | `true` | Crossfade transitions between all segments (FFmpeg `xfade`) |
| `crossfade_duration` | `0.5` | Crossfade length in seconds |
| `music_fade_out` | `0` | Seconds of music fade-out at the end of the video (0 = none) |
| `intro` / `outro` | — | Optional branded logo segments; omit the key to skip |

### Intro / Outro Segments

`intro` and `outro` render a logo centred on a solid background, before and after the slides:

| Field | Default | Description |
|-------|---------|-------------|
| `image` | `logo.png` | Logo file, resolved from the job's `images/` directory (or its root) |
| `duration` | `3` | Segment length in seconds |
| `background` | `white` | Background colour (any FFmpeg colour name or hex) |

Intro and outro segments carry no narration, but the audio timeline accounts for them —
narration is offset by the intro length, and total duration subtracts the crossfade overlaps.
If the logo file is missing, the segment is silently skipped.

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

| Variable | CPU Default | GPU Default | Description |
|----------|-------------|-------------|-------------|
| `API_PORT` | `8189` | `8189` | Host port (container always listens on 8080) |
| `MAX_STORED_JOBS` | `20` | `20` | Completed/failed jobs kept before pruning |
| `DATA_PATH` | `./data` | `./data` | Host path for job storage (`/data`) |
| `MODELS_PATH` | `./models` | `./models` | Host path for the model cache (`/models`) |
| `DEVICE` | `cpu` | `cuda` | Fixed per compose file |
| `TTS_DEVICE` | `cpu` | `cuda` | Fixed per compose file |
| `MUSICGEN_MODEL` | `facebook/musicgen-small` | `facebook/musicgen-large` | MusicGen checkpoint |
| `FFMPEG_ENCODER` | `libx264` | `h264_nvenc` | Video encoder |
| `NUM_THREADS` | `12` | `24` | Match your P-core count |
| `MUSIC_VOLUME` | `0.12` | `0.12` | Music level relative to narration (0.0–1.0) |
| `HF_TOKEN` | (optional) | (optional) | HuggingFace token for gated models |
| `INPUT_PATH` / `OUTPUT_PATH` | `./input` / `./output` | same | Legacy CLI mode volumes only |

## Pipeline Stages

Stage order is fixed and identical in both front-ends (`server.py` and `entrypoint.sh`):

1. **Narration Synthesis** — Per-slide TTS (Chatterbox / IndicF5 by language). Runs **first** because it writes `durations.json`, which determines every slide's on-screen time
2. **Image Preparation** — Normalise to target resolution via ImageMagick
3. **Music Generation** — Background music via MusicGen (30s clips, auto-stitched)
4. **Audio Mixing** — Combine narration + music with FFmpeg, honouring intro/outro offsets and crossfade overlap
5. **Video Segments + Overlays** — Render per-slide text overlays, plus branded intro/outro segments
6. **Final Assembly** — Concatenate segments (xfade crossfades) + mixed audio → MP4

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
