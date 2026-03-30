#!/usr/bin/env python3
"""Render audio from the optimizer's saved state.

Usage:
    python render.py                        # render to analysis/rendered.wav
    python render.py -o output.wav          # render to custom path
    python render.py --defaults             # render from PARAM_DEFAULTS instead
"""

import argparse, json, os, sys
import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build'))

from optimize_full import (
    PARAM_SPEC, PARAM_DEFAULTS, SR,
    unpack_params, render_with_params,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(ROOT, 'analysis', 'process', 'optimizer_state.json')


def load_from_state():
    """Load params and patterns from optimizer state."""
    if not os.path.exists(STATE_PATH):
        print(f"ERROR: {STATE_PATH} not found. Run optimize.py first.")
        sys.exit(1)

    with open(STATE_PATH) as f:
        state = json.load(f)

    x = PARAM_DEFAULTS.copy()
    for i, (name, _, _, _) in enumerate(PARAM_SPEC):
        if name in state.get('params', {}):
            x[i] = state['params'][name]

    patterns = []
    for pat_dict in state.get('patterns', []):
        pat = {}
        for key, vel in pat_dict.items():
            r, c = key.split('-')
            pat[(int(r), int(c))] = vel
        patterns.append(pat)

    composite = state.get('composite', '?')
    iteration = state.get('iteration', '?')
    return x, patterns, iteration, composite


def main():
    parser = argparse.ArgumentParser(description='Render audio from optimizer state')
    parser.add_argument('-o', '--output', default=None, help='Output .wav path')
    parser.add_argument('--defaults', action='store_true',
                        help='Render from PARAM_DEFAULTS instead of saved state')
    args = parser.parse_args()

    out_path = args.output or os.path.join(ROOT, 'analysis', 'rendered.wav')

    if args.defaults:
        from render_original import PATTERNS
        x = PARAM_DEFAULTS.copy()
        patterns = PATTERNS
        print(f"Rendering from PARAM_DEFAULTS...")
    else:
        x, patterns, iteration, composite = load_from_state()
        print(f"Rendering from optimizer state (iteration {iteration}, composite={composite})...")

    p = unpack_params(x)
    print(f"  Frequencies: {[round(f, 1) for f in p['freqs']]}")
    print(f"  Step ms: {p['step_ms']:.1f}")
    print(f"  Patterns: {len(patterns)}")

    # Render at full sample rate (stereo)
    print(f"  Rendering at {SR}Hz...")
    mono = render_with_params(x, patterns, fast=False)

    # Convert to stereo 16-bit WAV
    stereo = np.column_stack([mono, mono])
    sig16 = np.clip(stereo, -1, 1)
    sig16 = (sig16 * 32767).astype(np.int16)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    wavfile.write(out_path, SR, sig16)

    duration = len(mono) / SR
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Written: {out_path} ({duration:.1f}s, {size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
