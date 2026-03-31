#!/bin/bash
set -e

# Polynome optimizer setup: extract reference audio from video, then run optimizer.
#
# Usage:
#   ./setup.sh           # extract + prepare + optimize
#   ./setup.sh --no-opt  # just extract + prepare (skip optimizer)

ROOT="$(cd "$(dirname "$0")" && pwd)"
VIDEO="$ROOT/analysis/original/Monome video demo.mp4"
RAW_WAV="$ROOT/analysis/reference_raw.wav"
REF_WAV="$ROOT/analysis/reference.wav"

# Step 1: Extract audio from mp4
if [ ! -f "$REF_WAV" ]; then
    echo "=== Step 1: Extract audio from video ==="
    if [ ! -f "$VIDEO" ]; then
        echo "ERROR: Video not found: $VIDEO"
        exit 1
    fi

    if [ ! -f "$RAW_WAV" ]; then
        echo "Extracting audio from: $VIDEO (2:58 - 4:50, original mode segment)"
        ffmpeg -i "$VIDEO" -ss 178 -to 290 -vn -acodec pcm_s16le -ar 44100 -ac 1 "$RAW_WAV" -y -loglevel warning
        echo "  -> $RAW_WAV"
    else
        echo "Raw audio already exists: $RAW_WAV"
    fi

    # Step 2: Prepare reference (trim, filter, normalize)
    echo ""
    echo "=== Step 2: Prepare reference audio ==="
    python3 "$ROOT/build/prepare_reference.py" --input "$RAW_WAV" --output "$REF_WAV"
    echo "  -> $REF_WAV"
else
    echo "Reference audio already exists: $REF_WAV"
fi

# Step 3: Run optimizer
if [ "$1" != "--no-opt" ]; then
    echo ""
    echo "=== Step 3: Run optimizer ==="
    echo "Press Ctrl+C to stop gracefully."
    echo ""
    python3 "$ROOT/optimize.py" "$@"
fi
