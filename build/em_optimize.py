#!/usr/bin/env python3
"""EM-style optimization of synthesis parameters.

Given the reference audio, iteratively:
  E-step: Infer timeline (onset times → button presses)
  M-step: Optimize synthesis parameters to minimize per-note spectral distance

Usage:
    python build/em_optimize.py [--ref analysis/reference.wav] [--iterations 5]
"""

import argparse, json, os, sys
import numpy as np
import librosa
from scipy.optimize import minimize

SR = 44100
DEFAULT_FREQS = [587, 494, 415, 349, 294, 247, 196, 147]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def isolate_notes(y, sr):
    """Detect and isolate individual note events from reference audio."""
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='samples', backtrack=True)
    notes = []
    for i, onset in enumerate(onsets):
        # Extract 400ms segment
        end = min(onset + int(sr * 0.4), len(y))
        segment = y[onset:end]
        if len(segment) < sr * 0.03:
            continue

        # Detect pitch
        try:
            f0, _, vp = librosa.pyin(segment[:int(sr * 0.1)], fmin=80, fmax=1500, sr=sr)
            f0_valid = f0[~np.isnan(f0)]
            if len(f0_valid) == 0:
                continue
            freq = float(np.median(f0_valid))
        except Exception:
            continue

        # Map to row
        row = map_to_row(freq)
        if row < 0:
            continue

        # Compute spectral features
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Compute spectral envelope
        S = np.abs(librosa.stft(segment, n_fft=2048, hop_length=512))
        spec_mean = np.mean(S, axis=1)

        # Amplitude envelope
        env = np.abs(segment)
        win = max(1, int(sr * 0.002))
        env = np.convolve(env, np.ones(win)/win, mode='same')
        peak_idx = np.argmax(env)
        peak_val = env[peak_idx]
        attack_ms = peak_idx / sr * 1000

        # Decay: time to -20dB
        if peak_val > 1e-10:
            threshold = peak_val * 0.1
            below = np.where(env[peak_idx:] < threshold)[0]
            decay_s = below[0] / sr if len(below) > 0 else (len(segment) - peak_idx) / sr
        else:
            decay_s = 0.3

        notes.append({
            'onset_sample': int(onset),
            'onset_time': float(onset / sr),
            'freq': freq,
            'row': row,
            'segment': segment,
            'mfcc_mean': mfcc_mean,
            'spec_mean': spec_mean,
            'attack_ms': float(attack_ms),
            'decay_s': float(decay_s),
            'peak': float(peak_val),
        })

    return notes


def map_to_row(freq, freqs=DEFAULT_FREQS, tolerance_cents=150):
    """Map frequency to nearest row index."""
    best_row = -1
    best_dist = float('inf')
    for row, ref_freq in enumerate(freqs):
        for octave in [-1, 0, 1]:
            shifted = ref_freq * (2 ** octave)
            if shifted < 40:
                continue
            dist = abs(1200 * np.log2(max(freq, 1) / shifted))
            if dist < best_dist:
                best_dist = dist
                best_row = row
    if best_dist > tolerance_cents:
        return -1
    return best_row


def compute_reference_targets(notes):
    """Compute per-row average spectral characteristics from reference notes."""
    from collections import defaultdict
    rows = defaultdict(list)
    for note in notes:
        rows[note['row']].append(note)

    targets = {}
    for row, row_notes in rows.items():
        n = len(row_notes)
        if n == 0:
            continue

        # Average MFCC
        avg_mfcc = np.mean([note['mfcc_mean'] for note in row_notes], axis=0)

        # Average spectral envelope
        max_len = max(len(note['spec_mean']) for note in row_notes)
        specs = []
        for note in row_notes:
            padded = np.zeros(max_len)
            padded[:len(note['spec_mean'])] = note['spec_mean']
            specs.append(padded)
        avg_spec = np.mean(specs, axis=0)

        # Average envelope params
        avg_attack = np.median([note['attack_ms'] for note in row_notes])
        avg_decay = np.median([note['decay_s'] for note in row_notes])
        avg_peak = np.median([note['peak'] for note in row_notes])

        targets[row] = {
            'n_notes': n,
            'mfcc_mean': avg_mfcc,
            'spec_mean': avg_spec,
            'attack_ms': float(avg_attack),
            'decay_s': float(avg_decay),
            'peak': float(avg_peak),
            'freq': DEFAULT_FREQS[row],
        }
        print(f"  Row {row} ({DEFAULT_FREQS[row]}Hz): {n} notes, "
              f"attack={avg_attack:.1f}ms, decay={avg_decay:.3f}s")

    return targets


def synthesize_test_note(freq, params, sr=SR):
    """Synthesize a single note with given parameters and return its MFCC."""
    from common import (sine, noise, env_exp_decay, bandpass, comb_filter,
                        asymmetric_saturate, normalize, fade_in, fade_out,
                        highpass)

    dur = 0.4
    n = int(sr * dur)
    t = np.arange(n) / sr

    # Attack transient
    atk_dur = 0.008
    atk_n = int(sr * atk_dur)
    atk = noise(atk_dur, sr)
    atk = bandpass(atk, max(20, freq * 0.5), min(sr / 2 - 100, freq * 4), sr)
    atk_env = env_exp_decay(atk_dur, 0.2, 0.003, sr)
    atk *= atk_env * params['attack_level']

    # Tonal body
    sig = np.zeros(n)
    for h in range(len(params['harmonic_amps_db'])):
        partial_freq = freq * (h + 1)
        if partial_freq >= sr / 2:
            break
        amp = 10 ** (params['harmonic_amps_db'][h] / 20)
        sig += amp * np.sin(2 * np.pi * partial_freq * t)

    # Body resonance
    body_exc = np.zeros(n)
    body_exc[:atk_n] = atk[:min(atk_n, len(atk))]
    delay = max(1, int(sr / freq))
    body = comb_filter(body_exc, delay, feedback=params['comb_feedback'],
                       lp_freq=min(freq * 3, sr / 2 - 100), sr=sr)
    body *= params['comb_mix']
    sig = sig + body[:n]

    # Envelope
    env = env_exp_decay(dur, params['attack_ms'], params['decay_time'], sr)
    sig *= env

    # Insert attack
    sig[:len(atk)] += atk[:min(len(atk), n)]

    # Saturation
    sig = asymmetric_saturate(sig, drive=params['drive'], asymmetry=params['asymmetry'])

    # HF boost
    sig_hp = highpass(sig, 1500, sr) * params['hf_boost']
    sig = sig + sig_hp

    sig = normalize(sig, 0.85)
    fade_in(sig, params['attack_ms'], sr)
    fade_out(sig, 40, sr)

    return sig


def compute_note_mfcc(sig, sr=SR):
    """Compute mean MFCC of a synthesized note."""
    mfcc = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)


def optimize_sound_params(targets, current_params):
    """M-step: Optimize synthesis parameters to match reference per-note characteristics."""

    # We optimize a reduced set of parameters that have the most impact on MFCC
    param_names = ['attack_ms', 'decay_time', 'drive', 'hf_boost', 'comb_feedback', 'comb_mix']

    # Weighted combination of per-row MFCC distances
    def objective(x):
        params = current_params.copy()
        params['attack_ms'] = max(0.5, x[0])
        params['decay_time'] = max(0.05, x[1])
        params['drive'] = max(1.0, x[2])
        params['hf_boost'] = max(0.0, x[3])
        params['comb_feedback'] = np.clip(x[4], 0.0, 0.95)
        params['comb_mix'] = max(0.0, x[5])

        total_dist = 0
        total_weight = 0
        for row, target in targets.items():
            freq = target['freq']
            try:
                sig = synthesize_test_note(freq, params)
                mfcc = compute_note_mfcc(sig)
                # MFCC distance
                dist = np.linalg.norm(mfcc - target['mfcc_mean'])
                total_dist += dist * target['n_notes']
                total_weight += target['n_notes']
            except Exception as e:
                total_dist += 100 * target['n_notes']
                total_weight += target['n_notes']

        return total_dist / max(total_weight, 1)

    x0 = [
        current_params['attack_ms'],
        current_params['decay_time'],
        current_params['drive'],
        current_params['hf_boost'],
        current_params['comb_feedback'],
        current_params['comb_mix'],
    ]

    bounds = [
        (0.5, 10.0),    # attack_ms
        (0.05, 1.0),    # decay_time
        (1.0, 3.0),     # drive
        (0.0, 5.0),     # hf_boost
        (0.0, 0.95),    # comb_feedback
        (0.0, 0.5),     # comb_mix
    ]

    print(f"\n  Starting optimization (objective at x0: {objective(x0):.2f})")

    result = minimize(objective, x0, method='Nelder-Mead',
                     options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.5})

    optimized = current_params.copy()
    optimized['attack_ms'] = max(0.5, result.x[0])
    optimized['decay_time'] = max(0.05, result.x[1])
    optimized['drive'] = max(1.0, result.x[2])
    optimized['hf_boost'] = max(0.0, result.x[3])
    optimized['comb_feedback'] = np.clip(result.x[4], 0.0, 0.95)
    optimized['comb_mix'] = max(0.0, result.x[5])

    print(f"  Optimized objective: {result.fun:.2f}")
    print(f"  Parameters:")
    for name in param_names:
        print(f"    {name}: {current_params[name]:.3f} → {optimized[name]:.3f}")

    return optimized, float(result.fun)


def optimize_harmonics(targets, base_params):
    """Optimize harmonic amplitudes to match reference spectral profile."""
    n_harmonics = 10

    def objective(amps_db):
        params = base_params.copy()
        params['harmonic_amps_db'] = list(amps_db)

        total_dist = 0
        total_weight = 0
        for row, target in targets.items():
            freq = target['freq']
            try:
                sig = synthesize_test_note(freq, params)
                mfcc = compute_note_mfcc(sig)
                dist = np.linalg.norm(mfcc - target['mfcc_mean'])
                total_dist += dist * target['n_notes']
                total_weight += target['n_notes']
            except Exception:
                total_dist += 100 * target['n_notes']
                total_weight += target['n_notes']

        return total_dist / max(total_weight, 1)

    x0 = np.array(base_params['harmonic_amps_db'][:n_harmonics])

    print(f"\n  Optimizing harmonics (start: {objective(x0):.2f})")

    result = minimize(objective, x0, method='Nelder-Mead',
                     options={'maxiter': 300, 'xatol': 0.5, 'fatol': 0.5})

    optimized_db = [round(float(x), 1) for x in result.x]
    # Force fundamental to be 0 dB
    offset = optimized_db[0]
    optimized_db = [round(x - offset, 1) for x in optimized_db]

    print(f"  Optimized objective: {result.fun:.2f}")
    print(f"  Harmonics dB: {base_params['harmonic_amps_db'][:n_harmonics]}")
    print(f"  → Optimized:  {optimized_db}")

    return optimized_db, float(result.fun)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', default=None)
    parser.add_argument('--iterations', type=int, default=3)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref_path = args.ref or os.path.join(root, 'analysis', 'reference.wav')

    if not os.path.exists(ref_path):
        print(f"ERROR: {ref_path} not found")
        sys.exit(1)

    print(f"Loading: {ref_path}")
    y, sr = librosa.load(ref_path, sr=SR, mono=True)
    print(f"  Duration: {len(y)/sr:.1f}s")

    # E-step: Isolate notes and detect characteristics
    print("\n[E-step] Isolating notes from reference...")
    notes = isolate_notes(y, sr)
    print(f"  Found {len(notes)} notes")

    print("\n[E-step] Computing per-row targets...")
    targets = compute_reference_targets(notes)

    if not targets:
        print("ERROR: No targets found")
        sys.exit(1)

    # Current parameters (from gen_original.py)
    current_params = {
        'attack_ms': 1.5,
        'decay_time': 0.45,
        'drive': 2.0,
        'asymmetry': 0.05,
        'hf_boost': 2.0,
        'comb_feedback': 0.0,
        'comb_mix': 0.0,
        'attack_level': 0.35,
        'harmonic_amps_db': [0, -3, -6, -10, -14, -18, -22, -26, -32, -38],
    }

    for iteration in range(args.iterations):
        print(f"\n{'='*60}")
        print(f"EM Iteration {iteration + 1}")
        print(f"{'='*60}")

        # M-step 1: Optimize envelope/effects parameters
        print("\n[M-step 1] Optimizing envelope/effects...")
        optimized_params, obj1 = optimize_sound_params(targets, current_params)

        # M-step 2: Optimize harmonics with the new envelope params
        print("\n[M-step 2] Optimizing harmonics...")
        opt_harmonics, obj2 = optimize_harmonics(targets, optimized_params)
        optimized_params['harmonic_amps_db'] = opt_harmonics

        current_params = optimized_params
        print(f"\n  Iteration {iteration + 1} complete. Objectives: env={obj1:.2f}, harm={obj2:.2f}")

    # Output recommended parameters
    print(f"\n{'='*60}")
    print("RECOMMENDED PARAMETERS")
    print(f"{'='*60}")
    print(f"attack_ms: {current_params['attack_ms']:.2f}")
    print(f"decay_time: {current_params['decay_time']:.3f}")
    print(f"drive: {current_params['drive']:.3f}")
    print(f"hf_boost: {current_params['hf_boost']:.3f}")
    print(f"comb_feedback: {current_params['comb_feedback']:.3f}")
    print(f"comb_mix: {current_params['comb_mix']:.3f}")
    print(f"attack_level: {current_params['attack_level']:.3f}")
    print(f"harmonic_amps_db: {current_params['harmonic_amps_db']}")

    # Save results
    out_path = os.path.join(root, 'analysis', 'process', 'em_optimized_params.json')
    save_params = {k: v if not isinstance(v, np.ndarray) else v.tolist()
                   for k, v in current_params.items()}
    with open(out_path, 'w') as f:
        json.dump(save_params, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
