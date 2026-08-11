#!/usr/bin/env bash
# Run one render+compare iteration.
# Usage: bash build/iterate.sh <iteration_number> "description of change"

set -euo pipefail

ITER="${1:?Usage: iterate.sh <iteration> \"description\"}"
DESC="${2:?Usage: iterate.sh <iteration> \"description\"}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Iteration $ITER: $DESC ==="
echo ""

echo "--- Rendering ---"
python3 "$ROOT/build/render_original.py"
echo ""

echo "--- Comparing ---"
python3 "$ROOT/build/compare_audio.py" --iteration "$ITER" --description "$DESC"
