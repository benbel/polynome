#!/usr/bin/env python3
"""Full parameterized optimizer for original mode audio.

Parameterizes the entire rendering pipeline (synthesis, effects, patterns, timing)
and uses EM-style optimization to minimize composite distance to the reference.

E-step: Given current sound model, infer button press timeline from reference audio
M-step: Given button presses, optimize all synthesis/effects parameters

Usage:
    python build/optimize_full.py [--ref analysis/reference.wav] [--em-iters 3]
"""

import argparse, json, os, sys, time
import numpy as np
import librosa
from scipy.optimize import minimize, differential_evolution
from scipy.io import wavfile
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    SR, sine, noise, env_exp_decay,
    lowpass, highpass, bandpass, comb_filter,
    asymmetric_saturate, normalize, fade_in, fade_out,
    to_stereo, generate_reverb_ir,
)

# ── Column sequences (fixed from amxd) ──────────────────────────────────────
COL_SEQS = [
    [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
    [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
    [1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1],
    [1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1],
    [1,1,0,1,1,0,1,0,1,0,1,0,1,0,1,0],
    [1,0,1,0,1,1,0,1,0,1,0,1,0,1,0,1],
    [1,1,0,0,1,0,1,0,0,0,1,0,1,0,0,1],
    [1,0,1,1,0,1,0,1,0,0,0,1,0,1,0,0],
    [1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0],
    [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0],
    [1,0,1,0,0,1,0,1,0,0,1,0,0,1,0,0],
    [1,0,0,0,1,0,1,0,0,1,0,0,1,0,1,0],
    [0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0],
    [1,0,0,1,0,1,0,0,1,0,0,0,1,0,0,1],
    [0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0],
    [1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0],
]

STEP_VELS = [127,36,64,36,127,36,64,36,127,36,72,36,127,36,90,36]
STEP_VELS = [v / 127 for v in STEP_VELS]
SEQ_LEN = 16
STEPS_PER_PATTERN = 64
COMPARE_SR = 22050  # sample rate used by compare_audio.py


# ══════════════════════════════════════════════════════════════════════════════
# Parameter packing/unpacking
# ══════════════════════════════════════════════════════════════════════════════

PARAM_SPEC = [
    # Synthesis params
    ('attack_ms',       1.5,    0.3,   15.0),
    ('decay_time',      0.45,   0.05,   1.5),
    ('drive',           2.0,    1.0,    4.0),
    ('asymmetry',       0.05,   0.0,    0.3),
    ('hf_boost',        2.0,    0.0,    5.0),
    ('comb_feedback',   0.0,    0.0,    0.95),
    ('comb_mix',        0.0,    0.0,    0.5),
    ('attack_level',    0.35,   0.0,    1.0),
    # Harmonic amplitudes (dB relative to fundamental)
    ('harm_1_db',       -3.0,  -30.0,   0.0),
    ('harm_2_db',       -6.0,  -30.0,   0.0),
    ('harm_3_db',      -10.0,  -40.0,   0.0),
    ('harm_4_db',      -14.0,  -40.0,   0.0),
    ('harm_5_db',      -18.0,  -40.0,   0.0),
    ('harm_6_db',      -22.0,  -50.0,   0.0),
    ('harm_7_db',      -26.0,  -50.0,   0.0),
    ('harm_8_db',      -32.0,  -60.0,   0.0),
    ('harm_9_db',      -38.0,  -60.0,   0.0),
    # Effects
    ('reverb_wet',      0.08,   0.0,    0.5),
    ('reverb_length',   1.0,    0.2,    3.0),
    ('reverb_dark',     0.35,   0.0,    0.9),
    ('delay_wet',       0.15,   0.0,    0.5),
    ('delay_feedback',  0.25,   0.0,    0.7),
    ('delay_dark_lp',   3000,   500,   8000),
    # Timing & level
    ('step_ms',         74.0,   50.0,  120.0),
    ('target_rms',      0.30,   0.10,   0.60),
    # Frequencies (8 rows)
    ('freq_0',          587,    200,    800),
    ('freq_1',          494,    150,    700),
    ('freq_2',          415,    130,    600),
    ('freq_3',          349,    110,    500),
    ('freq_4',          294,    90,     420),
    ('freq_5',          247,    80,     350),
    ('freq_6',          196,    60,     280),
    ('freq_7',          147,    50,     220),
]

PARAM_NAMES = [p[0] for p in PARAM_SPEC]
PARAM_DEFAULTS = np.array([p[1] for p in PARAM_SPEC])
PARAM_BOUNDS = [(p[2], p[3]) for p in PARAM_SPEC]


def unpack_params(x):
    """Convert parameter vector to named dict."""
    p = {}
    for i, (name, _, _, _) in enumerate(PARAM_SPEC):
        p[name] = float(x[i])

    # Reconstruct harmonic_amps_db list (fundamental is always 0 dB)
    p['harmonic_amps_db'] = [0.0] + [p[f'harm_{i}_db'] for i in range(1, 10)]
    p['freqs'] = [p[f'freq_{i}'] for i in range(8)]
    return p


# ══════════════════════════════════════════════════════════════════════════════
# Parameterized synthesis
# ══════════════════════════════════════════════════════════════════════════════

def gen_tone_param(freq, p, sr=SR):
    """Generate a single tone with given parameters."""
    dur = 0.9
    n = int(sr * dur)
    t = np.arange(n) / sr

    # Attack transient
    atk_dur = 0.008
    atk_n = int(sr * atk_dur)
    atk = noise(atk_dur, sr)
    atk = bandpass(atk, max(20, freq * 0.5), min(sr/2 - 100, freq * 4), sr)
    atk_env = env_exp_decay(atk_dur, 0.2, 0.003, sr)
    atk *= atk_env * p['attack_level']

    # Tonal body
    sig = np.zeros(n)
    amps_db = p['harmonic_amps_db']
    for h in range(len(amps_db)):
        partial_freq = freq * (h + 1)
        if partial_freq >= sr / 2:
            break
        amp = 10 ** (amps_db[h] / 20)
        sig += amp * np.sin(2 * np.pi * partial_freq * t)

    # Body resonance via comb filter
    if p['comb_mix'] > 0.001:
        body_exc = np.zeros(n)
        body_exc[:atk_n] = atk[:min(atk_n, len(atk))]
        delay = max(1, int(sr / freq))
        body = comb_filter(body_exc, delay,
                          feedback=p['comb_feedback'],
                          lp_freq=min(freq * 3, sr/2 - 100), sr=sr)
        body *= p['comb_mix']
        sig = sig + body[:n]

    # Envelope
    env = env_exp_decay(dur, p['attack_ms'], p['decay_time'], sr)
    sig *= env

    # Insert attack
    sig[:len(atk)] += atk[:min(len(atk), n)]

    # Saturation
    sig = asymmetric_saturate(sig, drive=p['drive'], asymmetry=p['asymmetry'])

    # HF boost
    if p['hf_boost'] > 0.01:
        sig_hp = highpass(sig, 1500, sr) * p['hf_boost']
        sig = sig + sig_hp

    sig = normalize(sig, 0.85)
    fade_in(sig, p['attack_ms'], sr)
    fade_out(sig, 40, sr)
    return sig


# ══════════════════════════════════════════════════════════════════════════════
# Parameterized rendering
# ══════════════════════════════════════════════════════════════════════════════

def render_with_params(x, patterns, fast=False):
    """Full render pipeline with parameter vector x and given patterns.

    If fast=True, renders at half sample rate for speed.
    Returns mono audio at COMPARE_SR for comparison.
    """
    p = unpack_params(x)
    sr = SR // 2 if fast else SR

    # Pre-render samples
    freqs = p['freqs']
    samples = []
    for freq in freqs:
        sig = gen_tone_param(freq, p, sr)
        samples.append(sig)

    step_ms = p['step_ms']
    step_samples = int(step_ms / 1000 * sr)
    total_steps = len(patterns) * STEPS_PER_PATTERN
    total_samples = total_steps * step_samples + max(len(s) for s in samples)

    # Dry mix (stereo)
    mix = np.zeros((total_samples, 2))

    global_step = 0
    for pi, pattern in enumerate(patterns):
        for local_step in range(STEPS_PER_PATTERN):
            seq_idx = global_step % SEQ_LEN
            offset = global_step * step_samples

            for (row, col), vel in pattern.items():
                if col < len(COL_SEQS) and COL_SEQS[col][seq_idx]:
                    vol = vel * STEP_VELS[seq_idx]
                    sample = samples[row]
                    end = min(offset + len(sample), total_samples)
                    length = end - offset
                    mix[offset:end, 0] += sample[:length] * vol
                    mix[offset:end, 1] += sample[:length] * vol
            global_step += 1

    # Delay (simplified for speed in fast mode)
    if not fast:
        mix = apply_delay(mix, p, sr)

    # Reverb
    ir = generate_reverb_ir(p['reverb_length'], dark=p['reverb_dark'], sr=sr)
    for ch in range(2):
        wet = fftconvolve(mix[:, ch], ir[:, ch])[:mix.shape[0]]
        mix[:, ch] += wet * p['reverb_wet']

    # Normalize
    current_rms = np.sqrt(np.mean(mix ** 2))
    if current_rms > 0:
        gain = p['target_rms'] / current_rms
        mix *= gain
        mix = np.clip(mix, -0.95, 0.95)

    # Downmix to mono and resample to COMPARE_SR
    mono = np.mean(mix, axis=1)
    if sr != COMPARE_SR:
        from scipy.signal import resample
        n_out = int(len(mono) * COMPARE_SR / sr)
        mono = resample(mono, n_out)

    return mono


def apply_delay(mix, p, sr):
    """Stereo delay matching engine.js."""
    n = mix.shape[0]
    dl = int(0.33 * sr)
    dr = int(0.22 * sr)
    out = mix.copy()

    for ch, d in enumerate([dl, dr]):
        buf = np.zeros(n + d)
        buf[d:d+n] = mix[:, ch]
        for i in range(d, n + d):
            fb_idx = i - d
            if 0 <= fb_idx < n:
                buf[i] += p['delay_feedback'] * buf[fb_idx]
        delayed = buf[d:d+n]
        delayed = lowpass(delayed, p['delay_dark_lp'], sr)
        out[:, ch] += delayed * p['delay_wet']

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Distance computation (inline, matching compare_audio.py)
# ══════════════════════════════════════════════════════════════════════════════

def compute_composite(y_ren, y_ref, sr=COMPARE_SR):
    """Compute composite distance matching compare_audio.py methodology."""
    n = min(len(y_ref), len(y_ren))
    y_ref = y_ref[:n]
    y_ren = y_ren[:n]

    metrics = {}

    # Spectral centroid
    sc_ref = librosa.feature.spectral_centroid(y=y_ref, sr=sr)[0]
    sc_ren = librosa.feature.spectral_centroid(y=y_ren, sr=sr)[0]
    nc = min(len(sc_ref), len(sc_ren))
    metrics['spectral_centroid'] = float(np.mean(np.abs(sc_ref[:nc] - sc_ren[:nc])) / (sr / 2))

    # MFCC
    mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr, n_mfcc=13)
    mfcc_ren = librosa.feature.mfcc(y=y_ren, sr=sr, n_mfcc=13)
    dist = np.linalg.norm(np.mean(mfcc_ref, axis=1) - np.mean(mfcc_ren, axis=1))
    metrics['mfcc'] = min(dist / 200, 1.0)

    # Onset density
    onsets_ref = librosa.onset.onset_detect(y=y_ref, sr=sr)
    onsets_ren = librosa.onset.onset_detect(y=y_ren, sr=sr)
    rate_ref = len(onsets_ref) / max(len(y_ref)/sr, 0.1)
    rate_ren = len(onsets_ren) / max(len(y_ren)/sr, 0.1)
    if rate_ref == 0 and rate_ren == 0:
        metrics['onset_density'] = 0.0
    else:
        metrics['onset_density'] = abs(rate_ref - rate_ren) / max(rate_ref, rate_ren)

    # IOI histogram
    from scipy.stats import wasserstein_distance
    onsets_ref_t = librosa.onset.onset_detect(y=y_ref, sr=sr, units='time')
    onsets_ren_t = librosa.onset.onset_detect(y=y_ren, sr=sr, units='time')
    if len(onsets_ref_t) < 3 or len(onsets_ren_t) < 3:
        metrics['ioi_histogram'] = 1.0
    else:
        ioi_ref = np.diff(onsets_ref_t)
        ioi_ren = np.diff(onsets_ren_t)
        max_ioi = max(np.max(ioi_ref), np.max(ioi_ren), 0.01)
        metrics['ioi_histogram'] = min(wasserstein_distance(ioi_ref, ioi_ren) / max_ioi, 1.0)

    # RMS correlation
    fl = int(sr * 0.05)
    hl = fl // 2
    rms_ref = librosa.feature.rms(y=y_ref, frame_length=fl, hop_length=hl)[0]
    rms_ren = librosa.feature.rms(y=y_ren, frame_length=fl, hop_length=hl)[0]
    nr = min(len(rms_ref), len(rms_ren))
    if np.std(rms_ref[:nr]) < 1e-8 or np.std(rms_ren[:nr]) < 1e-8:
        metrics['rms_correlation'] = 1.0
    else:
        corr = np.corrcoef(rms_ref[:nr], rms_ren[:nr])[0, 1]
        metrics['rms_correlation'] = max(0, 1 - corr)

    # Pitch class
    ch_ref = librosa.feature.chroma_stft(y=y_ref, sr=sr)
    ch_ren = librosa.feature.chroma_stft(y=y_ren, sr=sr)
    hr = np.mean(ch_ref, axis=1)
    hn = np.mean(ch_ren, axis=1)
    dot = np.dot(hr, hn)
    norm = np.linalg.norm(hr) * np.linalg.norm(hn)
    metrics['pitch_class'] = max(0, 1 - dot/norm) if norm > 1e-8 else 1.0

    # Spectral rolloff
    ro_ref = librosa.feature.spectral_rolloff(y=y_ref, sr=sr)[0]
    ro_ren = librosa.feature.spectral_rolloff(y=y_ren, sr=sr)[0]
    nro = min(len(ro_ref), len(ro_ren))
    metrics['spectral_rolloff'] = float(np.mean(np.abs(ro_ref[:nro] - ro_ren[:nro])) / (sr/2))

    weights = {
        'spectral_centroid': 0.15, 'mfcc': 0.25, 'onset_density': 0.10,
        'ioi_histogram': 0.10, 'rms_correlation': 0.15, 'pitch_class': 0.10,
        'spectral_rolloff': 0.15,
    }
    composite = sum(metrics[k] * weights[k] for k in weights)
    return composite, metrics


# ══════════════════════════════════════════════════════════════════════════════
# E-step: Infer button presses from reference audio
# ══════════════════════════════════════════════════════════════════════════════

def infer_buttons(y_ref, sr, freqs, step_ms):
    """Infer button press timeline from reference audio.

    Returns list of patterns, each a dict of (row, col) → velocity.
    """
    # Detect onsets and pitches
    onsets_samples = librosa.onset.onset_detect(y=y_ref, sr=sr, units='samples', backtrack=True)

    events = []
    for onset_s in onsets_samples:
        end = min(onset_s + int(sr * 0.15), len(y_ref))
        segment = y_ref[onset_s:end]
        if len(segment) < sr * 0.02:
            continue
        try:
            f0, _, vp = librosa.pyin(segment, fmin=80, fmax=1500, sr=sr)
            f0_valid = f0[~np.isnan(f0)]
            if len(f0_valid) == 0:
                continue
            freq = float(np.median(f0_valid))
        except Exception:
            continue

        # Map to row
        best_row, best_dist = -1, float('inf')
        for row, ref_freq in enumerate(freqs):
            for oct in [-1, 0, 1]:
                shifted = ref_freq * (2 ** oct)
                if shifted < 40:
                    continue
                dist = abs(1200 * np.log2(max(freq, 1) / shifted))
                if dist < best_dist:
                    best_dist = dist
                    best_row = row
        if best_dist > 150:
            continue

        onset_time = onset_s / sr
        global_step = int(round(onset_time / (step_ms / 1000)))
        events.append({
            'row': best_row,
            'time_s': float(onset_time),
            'global_step': global_step,
            'seq_idx': global_step % SEQ_LEN,
            'pattern_idx': global_step // STEPS_PER_PATTERN,
        })

    if not events:
        return []

    n_patterns = max(e['pattern_idx'] for e in events) + 1

    # Group by pattern
    patterns_events = [[] for _ in range(n_patterns)]
    for event in events:
        if 0 <= event['pattern_idx'] < n_patterns:
            patterns_events[event['pattern_idx']].append(event)

    # Infer active cells per pattern
    inferred = []
    cumulative_cells = set()

    for pi, pe in enumerate(patterns_events):
        row_seq_counts = {}
        for event in pe:
            key = (event['row'], event['seq_idx'])
            row_seq_counts[key] = row_seq_counts.get(key, 0) + 1

        rows_in_pattern = set(e['row'] for e in pe)
        pattern_cells = {}

        for row in rows_in_pattern:
            row_triggers = [s for (r, s), _ in row_seq_counts.items() if r == row]

            best_col, best_score = -1, -1
            for col in range(16):
                expected = set(s for s in range(SEQ_LEN) if COL_SEQS[col][s] == 1)
                observed = set(row_triggers)
                hits = len(observed & expected)
                misses = len(expected - observed)
                false_alarms = len(observed - expected)
                score = hits - 0.5 * misses - 1.0 * false_alarms
                if (row, col) in cumulative_cells:
                    score += 2.0
                if score > best_score:
                    best_score = score
                    best_col = col

            if best_col >= 0 and best_score > 0:
                pattern_cells[(row, best_col)] = 1

        for cell in cumulative_cells:
            if cell not in pattern_cells:
                pattern_cells[cell] = 1

        cumulative_cells.update(pattern_cells.keys())
        inferred.append(dict(pattern_cells))

    return inferred


# ══════════════════════════════════════════════════════════════════════════════
# Main optimization loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', default=None)
    parser.add_argument('--em-iters', type=int, default=3)
    parser.add_argument('--fast', action='store_true', help='Use half sample rate for speed')
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref_path = args.ref or os.path.join(root, 'analysis', 'reference.wav')

    if not os.path.exists(ref_path):
        print(f"ERROR: {ref_path} not found")
        sys.exit(1)

    # Load reference at comparison SR
    print(f"Loading reference: {ref_path}")
    y_ref_compare, _ = librosa.load(ref_path, sr=COMPARE_SR, mono=True)
    print(f"  Duration: {len(y_ref_compare)/COMPARE_SR:.1f}s")

    # Also load at full SR for onset/pitch detection
    y_ref_full, _ = librosa.load(ref_path, sr=SR, mono=True)

    # Start with current best parameters
    x = PARAM_DEFAULTS.copy()

    # Current patterns (from render_original.py)
    from render_original import PATTERNS as current_patterns

    patterns = current_patterns
    best_composite = float('inf')
    best_x = x.copy()
    best_patterns = patterns

    for em_iter in range(args.em_iters):
        print(f"\n{'='*70}")
        print(f"EM ITERATION {em_iter + 1}")
        print(f"{'='*70}")

        # ── E-step: Infer buttons ────────────────────────────────────────
        print("\n[E-step] Inferring button presses...")
        p = unpack_params(x)
        inferred = infer_buttons(y_ref_full, SR, p['freqs'], p['step_ms'])
        if inferred:
            print(f"  Inferred {len(inferred)} patterns")
            for i, pat in enumerate(inferred):
                cells = sorted(pat.keys())
                if cells:
                    print(f"    P{i:2d}: {cells}")

            # Test if inferred patterns are better
            print("  Testing inferred patterns vs current...")
            y_inferred = render_with_params(x, inferred, fast=args.fast)
            c_inferred, m_inferred = compute_composite(y_inferred, y_ref_compare)

            y_current = render_with_params(x, patterns, fast=args.fast)
            c_current, m_current = compute_composite(y_current, y_ref_compare)

            print(f"  Current patterns: {c_current:.4f}")
            print(f"  Inferred patterns: {c_inferred:.4f}")

            if c_inferred < c_current:
                print(f"  ✓ Using inferred patterns")
                patterns = inferred
            else:
                print(f"  × Keeping current patterns")
        else:
            print("  No patterns inferred, keeping current")

        # ── M-step: Optimize parameters ──────────────────────────────────
        print("\n[M-step] Optimizing synthesis parameters...")
        eval_count = [0]
        eval_best = [float('inf')]

        def objective(x_candidate):
            eval_count[0] += 1
            try:
                y_ren = render_with_params(x_candidate, patterns, fast=args.fast)
                composite, metrics = compute_composite(y_ren, y_ref_compare)
                if composite < eval_best[0]:
                    eval_best[0] = composite
                    if eval_count[0] % 5 == 0:
                        print(f"    eval {eval_count[0]}: {composite:.4f} (best so far)")
                return composite
            except Exception as e:
                print(f"    eval {eval_count[0]}: ERROR {e}")
                return 1.0

        # Initial evaluation
        c0 = objective(x)
        print(f"  Initial composite: {c0:.4f}")

        # Use Nelder-Mead for robustness with non-smooth landscape
        # Limit iterations since each eval is expensive
        result = minimize(
            objective, x,
            method='Nelder-Mead',
            options={
                'maxiter': 80,
                'maxfev': 120,
                'xatol': 0.001,
                'fatol': 0.001,
                'adaptive': True,
            }
        )

        print(f"\n  Optimization complete: {eval_count[0]} evaluations")
        print(f"  Best composite: {result.fun:.4f}")

        if result.fun < best_composite:
            best_composite = result.fun
            best_x = result.x.copy()
            best_patterns = patterns
            x = result.x.copy()
            print(f"  ✓ New best: {best_composite:.4f}")
        else:
            print(f"  × No improvement over previous best ({best_composite:.4f})")
            x = best_x.copy()

        # Print current best parameters
        p = unpack_params(best_x)
        print(f"\n  Current best parameters:")
        for name, val, _, _ in PARAM_SPEC:
            default = dict(zip(PARAM_NAMES, PARAM_DEFAULTS))[name]
            current = p[name] if name in p else dict(zip(PARAM_NAMES, best_x))[name]
            changed = " *" if abs(current - default) > 0.01 else ""
            print(f"    {name:20s}: {current:10.3f} (default: {default:.3f}){changed}")

    # ── Final evaluation with full quality ────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")

    # Re-evaluate at full quality if we were using fast mode
    y_final = render_with_params(best_x, best_patterns, fast=False)
    c_final, m_final = compute_composite(y_final, y_ref_compare)
    print(f"\nFinal composite distance: {c_final:.4f}")
    print(f"\nPer-metric breakdown:")
    for k, v in sorted(m_final.items()):
        print(f"  {k:25s}: {v:.4f}")

    # Save optimized parameters
    p = unpack_params(best_x)
    out = {
        'composite': c_final,
        'metrics': {k: round(v, 6) for k, v in m_final.items()},
        'params': {name: round(float(best_x[i]), 4) for i, (name, _, _, _) in enumerate(PARAM_SPEC)},
        'patterns': [
            {f"{r}-{c}": v for (r, c), v in pat.items()}
            for pat in best_patterns
        ],
    }
    out_path = os.path.join(root, 'analysis', 'process', 'optimized_full.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print code to apply
    print(f"\n{'='*70}")
    print("CODE TO APPLY (gen_original.py DEFAULT_ENVELOPE):")
    print(f"{'='*70}")
    print(f"DEFAULT_ENVELOPE = {{")
    print(f"    'attack_ms': {p['attack_ms']:.2f},")
    print(f"    'decay_time': {p['decay_time']:.3f},")
    print(f"    'harmonic_ratios': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0],")
    print(f"    'harmonic_amplitudes_db': {[round(x, 1) for x in p['harmonic_amps_db']]},")
    print(f"    'inharmonicity_cents': [0, 6, -2, 10, -4, 5, -7, 8, -3, 6],")
    print(f"}}")
    print(f"\ngen_tone drive={p['drive']:.2f}, asymmetry={p['asymmetry']:.3f}")
    print(f"hf_boost={p['hf_boost']:.2f}, comb_feedback={p['comb_feedback']:.3f}, comb_mix={p['comb_mix']:.3f}")
    print(f"attack_level={p['attack_level']:.3f}")
    print(f"\nEffects:")
    print(f"  reverb_wet={p['reverb_wet']:.3f}, reverb_length={p['reverb_length']:.2f}, reverb_dark={p['reverb_dark']:.3f}")
    print(f"  delay_wet={p['delay_wet']:.3f}, delay_feedback={p['delay_feedback']:.3f}, delay_dark_lp={p['delay_dark_lp']:.0f}")
    print(f"  step_ms={p['step_ms']:.1f}, target_rms={p['target_rms']:.3f}")
    print(f"\nFrequencies: {[round(f, 1) for f in p['freqs']]}")


if __name__ == '__main__':
    main()
