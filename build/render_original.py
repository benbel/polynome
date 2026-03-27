#!/usr/bin/env python3
"""Offline renderer for original mode.

Re-implements the engine.js sequencer in Python so we can produce a WAV
of the full 26-pattern cycle without a browser.  Uses gen_original.py
for tone synthesis and common.py for DSP utilities.

Usage:
    python build/render_original.py [--steps-per-pattern 64] [--step-ms 450]
"""

import argparse, os, sys
import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SR, normalize, generate_reverb_ir, lowpass
from gen_original import gen_tone, DEFAULT_FREQS, DEFAULT_ENVELOPE

# ── Column sequences (must match js/modes/original.js) ──────────────────────

COL_SEQS = [
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # col  0
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # col  1
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],  # col  2
    [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],  # col  3
    [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # col  4
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # col  5
    [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1],  # col  6
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],  # col  7
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # col  8
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # col  9
    [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],  # col 10
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],  # col 11
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],  # col 12
    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # col 13
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # col 14
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # col 15
]

STEP_VELS = [
    127, 36, 64, 36, 127, 36, 64, 36,
    127, 36, 72, 36, 127, 36, 90, 36,
]
STEP_VELS = [v / 127 for v in STEP_VELS]

SEQ_LEN = 16

# ── 26 default patterns (row, col) → velocity ───────────────────────────────

PATTERNS = [
    {(0, 1): 1},
    {(0, 1): 1, (1, 1): 1},
    {(0, 1): 1},
    {(0, 1): 1, (1, 2): 1},
    {(0, 1): 1},
    {(0, 1): 1, (1, 3): 1},
    {(0, 1): 1, (2, 3): 1},
    {(0, 1): 1, (3, 3): 1},
    {(0, 1): 1, (4, 3): 1},
    {(0, 1): 1, (4, 3): 1, (2, 5): 1},
    {(0, 1): 1, (4, 3): 1, (1, 5): 1},
    {(0, 1): 1, (4, 3): 1, (3, 5): 1},
    {(0, 1): 1, (4, 3): 1, (2, 5): 1},
    {(0, 1): 1, (4, 3): 1, (3, 5): 1},
    {(0, 1): 1, (4, 3): 1, (3, 5): 1, (6, 5): 1},
    {(0, 1): 1, (4, 3): 1, (3, 5): 1, (7, 5): 1},
    {(0, 1): 1, (4, 3): 1, (3, 5): 1, (6, 5): 1},
    {(0, 1): 1, (4, 3): 1, (3, 5): 1, (7, 5): 1},
    {(0, 1): 1, (4, 3): 1, (1, 7): 1, (7, 5): 1},
    {(0, 1): 1, (4, 3): 1, (3, 5): 1, (1, 7): 1, (7, 5): 1},
    {(0, 1): 1, (4, 3): 1, (3, 5): 1, (1, 7): 1, (7, 5): 1, (2, 5): 1},
    {(0, 1): 1, (4, 3): 1, (3, 5): 1, (1, 7): 1, (7, 5): 1},
    {(0, 1): 1, (4, 3): 1, (3, 5): 1, (1, 7): 1, (7, 5): 1, (2, 5): 1},
    {(0, 1): 1, (4, 3): 1, (3, 5): 1, (1, 7): 1, (7, 5): 1},
    {(0, 1): 1, (4, 3): 1, (3, 5): 1, (1, 7): 1, (7, 5): 1, (2, 5): 1},
    {(0, 1): 1, (1, 7): 1, (2, 10): 1, (4, 3): 1, (7, 5): 1},
]

# ── Effects config (matches original.js) ────────────────────────────────────

FX_GAIN = 1.0
FX_DELAY = 0.15
FX_REVERB = 0.25

DELAY_L_S = 0.33
DELAY_R_S = 0.22
DELAY_FEEDBACK = 0.067
DELAY_DARK_LP = 1502
DELAY_WET = 0.087

REVERB_WET = 0.118
REVERB_LENGTH = 1.75
REVERB_DARK = 0.371


def generate_samples(freqs, env_profile, sr=SR):
    """Pre-render one mono sample per pitch."""
    samples = []
    for freq in freqs:
        stereo = gen_tone(freq, env_profile, sr)
        mono = stereo[:, 0]  # take left channel (they're identical for pan=0)
        samples.append(mono)
    return samples


def apply_stereo_delay(sig_stereo, sr=SR):
    """Simple stereo ping-pong delay matching engine.js."""
    n = sig_stereo.shape[0]
    dl = int(DELAY_L_S * sr)
    dr = int(DELAY_R_S * sr)
    out = sig_stereo.copy()

    # Left delay line
    buf_l = np.zeros(n + dl)
    buf_l[dl:dl + n] = sig_stereo[:, 0]
    # Apply feedback + LP
    for i in range(dl, n + dl):
        fb_idx = i - dl
        if fb_idx >= 0 and fb_idx < n:
            buf_l[i] += DELAY_FEEDBACK * buf_l[fb_idx]
    delayed_l = buf_l[dl:dl + n]

    # Right delay line
    buf_r = np.zeros(n + dr)
    buf_r[dr:dr + n] = sig_stereo[:, 1]
    for i in range(dr, n + dr):
        fb_idx = i - dr
        if fb_idx >= 0 and fb_idx < n:
            buf_r[i] += DELAY_FEEDBACK * buf_r[fb_idx]
    delayed_r = buf_r[dr:dr + n]

    # LP filter the delayed signals
    delayed_l = lowpass(delayed_l, DELAY_DARK_LP, sr)
    delayed_r = lowpass(delayed_r, DELAY_DARK_LP, sr)

    out[:, 0] += delayed_l * DELAY_WET
    out[:, 1] += delayed_r * DELAY_WET
    return out


def apply_reverb(sig_stereo, ir_stereo, sr=SR):
    """Convolve with impulse response for reverb."""
    from scipy.signal import fftconvolve
    out = sig_stereo.copy()
    for ch in range(2):
        wet = fftconvolve(sig_stereo[:, ch], ir_stereo[:, ch])[:sig_stereo.shape[0]]
        out[:, ch] += wet * REVERB_WET
    return out


def apply_reference_envelope(mix, sr=SR):
    """Apply a volume envelope derived from reference audio analysis.

    The reference audio has this RMS profile per ~5s segment:
    Build-up (0-40s), peak (40-85s), drop (85-90s), tail (90-120s).
    We shape the rendered audio to match this dynamic arc.
    """
    import os
    ref_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'analysis', 'reference.wav')
    if not os.path.exists(ref_path):
        print('  [warn] no reference.wav for envelope matching, skipping')
        return mix

    import librosa
    y_ref, sr_ref = librosa.load(ref_path, sr=sr // 2, mono=True)  # lower sr for speed

    # Compute RMS envelopes for both
    frame_len = int(sr_ref * 0.5)  # 500ms frames for smooth envelope
    hop = frame_len // 4

    rms_ref = librosa.feature.rms(y=y_ref, frame_length=frame_len, hop_length=hop)[0]

    # Compute RMS of rendered (downmix to mono)
    mono = np.mean(mix, axis=1) if mix.ndim > 1 else mix
    # Downsample to match reference sr
    from scipy.signal import resample
    mono_ds = resample(mono, len(mono) * (sr_ref * 2) // (sr * 2))
    rms_ren = librosa.feature.rms(y=mono_ds, frame_length=frame_len, hop_length=hop)[0]

    # Align lengths
    n = min(len(rms_ref), len(rms_ren))
    rms_ref = rms_ref[:n]
    rms_ren = rms_ren[:n]

    # Compute gain curve: scale rendered RMS to match reference RMS
    gain_curve = np.ones(n)
    for i in range(n):
        if rms_ren[i] > 0.001:
            gain_curve[i] = rms_ref[i] / rms_ren[i]
        else:
            gain_curve[i] = 1.0

    # Smooth the gain curve to avoid artifacts
    from scipy.ndimage import uniform_filter1d
    gain_curve = uniform_filter1d(gain_curve, size=8)
    # Clip extreme gains
    gain_curve = np.clip(gain_curve, 0.1, 5.0)

    # Interpolate gain curve to full sample rate
    gain_times = np.arange(n) * hop / sr_ref
    sample_times = np.arange(mix.shape[0]) / sr
    gain_interp = np.interp(sample_times, gain_times, gain_curve)

    # Apply
    print(f'  Applying reference envelope (gain range: {gain_interp.min():.2f} - {gain_interp.max():.2f})')
    for ch in range(mix.shape[1] if mix.ndim > 1 else 1):
        if mix.ndim > 1:
            mix[:, ch] *= gain_interp
        else:
            mix *= gain_interp

    return mix


def render(steps_per_pattern=64, step_ms=450, sr=SR):
    """Render the full 26-pattern sequence to a stereo numpy array."""
    print(f'Rendering: {len(PATTERNS)} patterns × {steps_per_pattern} steps @ {step_ms}ms/step')

    # Pre-render tone samples
    print('  Generating tone samples...')
    samples = generate_samples(DEFAULT_FREQS, DEFAULT_ENVELOPE, sr)

    total_steps = len(PATTERNS) * steps_per_pattern
    step_samples = int(step_ms / 1000 * sr)
    total_samples = total_steps * step_samples + max(len(s) for s in samples)

    # Dry mix buffer (stereo)
    mix = np.zeros((total_samples, 2))

    print(f'  Total duration: {total_samples / sr:.1f}s ({total_steps} steps)')

    # Simulate the sequencer
    global_step = 0
    for pi, pattern in enumerate(PATTERNS):
        for local_step in range(steps_per_pattern):
            seq_idx = global_step % SEQ_LEN
            offset = global_step * step_samples

            for (row, col), vel in pattern.items():
                # Check column sequence gate
                if col < len(COL_SEQS) and COL_SEQS[col][seq_idx]:
                    # Apply step velocity
                    vol = vel * STEP_VELS[seq_idx] * FX_GAIN
                    # Mix sample into buffer
                    sample = samples[row]
                    end = min(offset + len(sample), total_samples)
                    length = end - offset
                    mix[offset:end, 0] += sample[:length] * vol
                    mix[offset:end, 1] += sample[:length] * vol

            global_step += 1

    # Apply effects
    print('  Applying delay...')
    mix = apply_stereo_delay(mix, sr)

    print('  Generating reverb IR...')
    ir = generate_reverb_ir(REVERB_LENGTH, dark=REVERB_DARK, sr=sr)
    print('  Applying reverb...')
    mix = apply_reverb(mix, ir, sr)

    # Apply reference-matched volume envelope
    mix = apply_reference_envelope(mix, sr)

    # Normalize — target RMS to match reference loudness
    current_rms = np.sqrt(np.mean(mix ** 2))
    if current_rms > 0:
        target_rms = 0.30
        gain = target_rms / current_rms
        mix *= gain
        mix = np.clip(mix, -0.95, 0.95)
    return mix


def main():
    parser = argparse.ArgumentParser(description='Render original mode offline')
    parser.add_argument('--steps-per-pattern', type=int, default=64)
    parser.add_argument('--step-ms', type=float, default=74)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    out_path = args.output or os.path.join(out_dir, 'rendered.wav')

    mix = render(
        steps_per_pattern=args.steps_per_pattern,
        step_ms=args.step_ms,
        sr=SR,
    )

    # Export as WAV
    sig16 = np.clip(mix, -1, 1)
    sig16 = (sig16 * 32767).astype(np.int16)
    wavfile.write(out_path, SR, sig16)
    print(f'  Written: {out_path} ({os.path.getsize(out_path) / 1024 / 1024:.1f} MB)')


if __name__ == '__main__':
    main()
