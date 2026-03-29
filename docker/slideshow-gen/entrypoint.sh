#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────
# Slideshow Generator — Entrypoint
# Parses CLI args and runs the pipeline stages
# ─────────────────────────────────────────────────

# Defaults
INPUT="/input/script.json"
OUTPUT="/output/final.mp4"
MUSIC_PROMPT=""
MUSIC_DURATION=""
MUSIC_VOL="${MUSIC_VOLUME:-0.2}"
VOICE_REF=""
NO_MUSIC=false
NO_NARRATION=false
SUBTITLES=false
KEEP_WORKING=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)       INPUT="$2"; shift 2 ;;
        --input=*)     INPUT="${1#*=}"; shift ;;
        --output)      OUTPUT="$2"; shift 2 ;;
        --output=*)    OUTPUT="${1#*=}"; shift ;;
        --music-prompt)    MUSIC_PROMPT="$2"; shift 2 ;;
        --music-prompt=*)  MUSIC_PROMPT="${1#*=}"; shift ;;
        --music-duration)  MUSIC_DURATION="$2"; shift 2 ;;
        --music-duration=*)MUSIC_DURATION="${1#*=}"; shift ;;
        --music-volume)    MUSIC_VOL="$2"; shift 2 ;;
        --music-volume=*)  MUSIC_VOL="${1#*=}"; shift ;;
        --voice-ref)   VOICE_REF="$2"; shift 2 ;;
        --voice-ref=*) VOICE_REF="${1#*=}"; shift ;;
        --no-music)    NO_MUSIC=true; shift ;;
        --no-narration)NO_NARRATION=true; shift ;;
        --subtitles)   SUBTITLES=true; shift ;;
        --keep-working)KEEP_WORKING=true; shift ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

echo "═══════════════════════════════════════════════"
echo "  Slideshow Generator"
echo "  Device: ${DEVICE:-cpu} | Encoder: ${FFMPEG_ENCODER:-libx264}"
echo "═══════════════════════════════════════════════"
echo ""
echo "  Input:  ${INPUT}"
echo "  Output: ${OUTPUT}"
echo ""

# Validate input
if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: Script file not found: ${INPUT}" >&2
    exit 1
fi

WORKING="/tmp/slideshow-working"
mkdir -p "$WORKING"/{slides,narration,segments}

INPUT_DIR="$(dirname "$INPUT")"

# Export vars for Python scripts
export INPUT OUTPUT WORKING INPUT_DIR
export MUSIC_PROMPT MUSIC_DURATION MUSIC_VOL VOICE_REF
export NO_MUSIC NO_NARRATION SUBTITLES

TOTAL_START=$(date +%s)

# ── Stage 1: Image Preparation ──────────────────
echo "── Stage 1/6: Image Preparation ──"
python3 /app/scripts/stage_images.py "$INPUT" "$WORKING/slides"

# ── Stage 2: Narration Synthesis ─────────────────
if [[ "$NO_NARRATION" == "false" ]]; then
    echo "── Stage 2/6: Narration Synthesis ──"
    VOICE_ARG=""
    [[ -n "$VOICE_REF" ]] && VOICE_ARG="--voice-ref=$VOICE_REF"
    python3 /app/scripts/stage_narration.py "$INPUT" "$WORKING/narration" $VOICE_ARG
else
    echo "── Stage 2/6: Narration Synthesis [SKIPPED] ──"
fi

# ── Stage 3: Music Generation ────────────────────
if [[ "$NO_MUSIC" == "false" ]]; then
    echo "── Stage 3/6: Music Generation ──"
    PROMPT_ARG=""
    DURATION_ARG=""
    [[ -n "$MUSIC_PROMPT" ]] && PROMPT_ARG="--prompt=$MUSIC_PROMPT"
    [[ -n "$MUSIC_DURATION" ]] && DURATION_ARG="--duration=$MUSIC_DURATION"
    python3 /app/scripts/stage_music.py "$INPUT" "$WORKING/music.wav" $PROMPT_ARG $DURATION_ARG
else
    echo "── Stage 3/6: Music Generation [SKIPPED] ──"
fi

# ── Stage 4: Audio Mixing ────────────────────────
echo "── Stage 4/6: Audio Mixing ──"
python3 /app/scripts/stage_mix.py "$INPUT" "$WORKING" --volume="$MUSIC_VOL"

# ── Stage 5: Video Segments + Overlays ───────────
echo "── Stage 5/6: Video Segments + Overlays ──"
python3 /app/scripts/stage_segments.py "$INPUT" "$WORKING"

# ── Stage 6: Final Assembly ──────────────────────
echo "── Stage 6/6: Final Assembly ──"
SUBTITLE_ARG=""
[[ "$SUBTITLES" == "true" ]] && SUBTITLE_ARG="--subtitles"
python3 /app/scripts/stage_assemble.py "$INPUT" "$WORKING" "$OUTPUT" $SUBTITLE_ARG

TOTAL_END=$(date +%s)
ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "═══════════════════════════════════════════════"
echo "  ✅ Complete in ${ELAPSED}s"
echo "  Output: ${OUTPUT}"
[[ "$SUBTITLES" == "true" ]] && echo "  Subtitles: ${OUTPUT%.mp4}.srt"
echo "═══════════════════════════════════════════════"

# Cleanup
if [[ "$KEEP_WORKING" == "false" ]]; then
    rm -rf "$WORKING"
    echo "  Working files cleaned up."
else
    echo "  Working files kept at: ${WORKING}"
fi
