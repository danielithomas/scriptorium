# Kokoro TTS

[Kokoro](https://github.com/remsky/Kokoro-FastAPI) lightweight text-to-speech server with an OpenAI-compatible API. Runs on CPU with no GPU required.

## Features

- OpenAI TTS API-compatible endpoint (`/v1/audio/speech`)
- Multiple preset voices (British English focus)
- Low resource usage — runs comfortably on CPU
- No model downloads required (baked into the image)

## Quick Start

```bash
cp .env.example .env
docker compose up -d
```

## API

Kokoro implements the OpenAI TTS API format:

```bash
# Generate speech (OpenAI-compatible)
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "Hello, this is a test of the Kokoro TTS system.",
    "voice": "bf_emma"
  }' \
  --output speech.mp3

# List available voices
curl http://localhost:8880/v1/audio/voices
```

### Available Voices

Kokoro ships with several preset voices. Use the `/v1/audio/voices` endpoint to list them, or check the [Kokoro-FastAPI documentation](https://github.com/remsky/Kokoro-FastAPI).

Common voices include British English speakers prefixed with `bf_` (female) and `bm_` (male).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KOKORO_PORT` | `8880` | Host port for API |
| `TZ` | `UTC` | Container timezone |

## Notes

- No GPU required — designed for CPU inference.
- The Docker image includes all model weights (~500MB image size).
- Compatible with any client that supports the OpenAI TTS API format.
- For voice cloning or more expressive TTS, see the [chatterbox-tts](../chatterbox-tts/) stack.
