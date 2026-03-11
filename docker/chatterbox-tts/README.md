# Chatterbox TTS

[Chatterbox](https://github.com/devnen/chatterbox-v2) text-to-speech server with voice cloning support. Runs on CPU or NVIDIA GPU.

## Features

- **Voice cloning** from short reference audio clips (up to 30 seconds)
- Predefined voice library
- FastAPI REST API
- CPU and NVIDIA CUDA support
- HuggingFace model caching

## Quick Start

```bash
cp .env.example .env
# Edit .env — set RUNTIME=nvidia for GPU, or leave as cpu

# Create voice directories
mkdir -p voices reference_audio outputs logs

# Build and start
docker compose up -d --build
```

The web UI is available at `http://localhost:8004`.

## CPU vs GPU

| Mode | Build Arg | Config | Performance |
|------|-----------|--------|-------------|
| CPU | `RUNTIME=cpu` | `device: cpu` | ~2 min per paragraph |
| GPU | `RUNTIME=nvidia` | `device: cuda` | ~5 sec per paragraph |

To switch modes:
1. Update `RUNTIME` in `.env`
2. Update `tts_engine.device` in `config.yaml`
3. Rebuild: `docker compose up -d --build`

## Voice Cloning

Place reference audio files (WAV, MP3, OGG) in the `reference_audio/` directory. Files should be:
- 5–30 seconds of clear speech
- Single speaker, minimal background noise
- 24kHz+ sample rate recommended

## API Endpoints

```bash
# Generate speech with default voice
curl -X POST http://localhost:8004/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "voice_mode": "predefined", "voice_id": "Emily.wav"}'

# Generate speech with voice clone
curl -X POST http://localhost:8004/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello!", "voice_mode": "clone", "reference_audio": "my_voice.wav"}'
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_PORT` | `8004` | Host port for API/UI |
| `RUNTIME` | `cpu` | Build mode: `cpu` or `nvidia` |
| `HF_TOKEN` | *(empty)* | HuggingFace token for gated model access |

## Configuration

Edit `config.yaml` for detailed TTS settings:

- `tts_engine.device` — `cpu` or `cuda`
- `generation_defaults.temperature` — Voice variation (0.0–1.0)
- `generation_defaults.exaggeration` — Expressiveness (0.0–2.0)
- `generation_defaults.speed_factor` — Playback speed multiplier
- `audio_output.sample_rate` — Output sample rate (default 24000)

## Volumes

| Path | Description |
|------|-------------|
| `voices/` | Predefined voice WAV files |
| `reference_audio/` | Voice cloning reference clips |
| `outputs/` | Generated audio files (if `save_to_disk: true`) |
| `logs/` | Server logs |
| `hf_cache` | HuggingFace model cache (named volume) |

## Notes

- First startup downloads the Chatterbox model (~2GB). Subsequent starts use the cached version.
- The HuggingFace cache is a named volume to persist across container rebuilds.
- CPU mode is functional but slow. GPU is strongly recommended for production use.
