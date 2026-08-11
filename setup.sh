#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
VIDEO="$ROOT/analysis/original/Monome video demo.mp4"
RAW_WAV="$ROOT/analysis/reference_raw.wav"
REF_WAV="$ROOT/analysis/reference.wav"

if [ ! -f "$REF_WAV" ]; then
    echo "=== Step 1: Extract audio from video ==="
    if [ ! -f "$VIDEO" ]; then
        echo "ERROR: Video not found: $VIDEO"
        exit 1
    fi

    if [ ! -f "$RAW_WAV" ]; then
        echo "Extracting audio from: $VIDEO (2:58 - 4:26, original mode segment)"
        ffmpeg -i "$VIDEO" -ss 178 -to 266 -vn -acodec pcm_s16le -ar 44100 -ac 1 "$RAW_WAV" -y -loglevel warning
        echo "  -> $RAW_WAV"
    else
        echo "Raw audio already exists: $RAW_WAV"
    fi

    echo ""
    echo "=== Step 2: Prepare reference audio ==="
    python3 "$ROOT/build/prepare_reference.py" --input "$RAW_WAV" --output "$REF_WAV"
    echo "  -> $REF_WAV"
else
    echo "Reference audio already exists: $REF_WAV"
fi

if [ "$1" != "--no-opt" ]; then
    echo ""
    echo "=== Step 3: Run optimizer ==="
    echo "Press Ctrl+C to stop gracefully."
    echo ""
    python3 "$ROOT/optimize.py" "$@"
fi
