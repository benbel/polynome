#!/usr/bin/env python3
"""Analysis-resynthesis: extract tone characteristics from reference audio.

Isolates individual note events, analyzes their spectral content and envelope,
and derives synthesis parameters that match the reference timbre.

Usage:
    python build/analyze_tones.py [--ref analysis/reference.wav]
"""

import argparse, json, os, sys
import numpy as np
import librosa

SR = 44100
DEFAULT_FREQS = [587, 494, 415, 349, 294, 247, 196, 147]


def find_isolated_notes(y, sr, min_silence_ms=100):
    """Find note onsets with relatively clean attacks (minimal overlap)."""
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='samples', backtrack=True)
    onset_times = onsets / sr

    isolated = []
    for i, onset in enumerate(onsets):
        # Check if there's enough silence before this onset
        if i > 0:
            prev_end = onsets[i - 1] + int(sr * 0.1)  # prev note's initial transient
            gap_ms = (onset - prev_end) / sr * 1000
            if gap_ms < min_silence_ms:
                continue

        # Extract segment
        end = min(onset + int(sr * 0.5), len(y))
        segment = y[onset:end]
        if len(segment) < sr * 0.05:
            continue

        # Check SNR: peak of first 20ms vs last 50ms
        peak = np.max(np.abs(segment[:int(sr * 0.02)]))
        tail = np.sqrt(np.mean(segment[int(sr * 0.2):] ** 2)) if len(segment) > int(sr * 0.25) else 0
        snr = peak / max(tail, 1e-10)
        if snr < 3:
            continue

        isolated.append({
            'onset_sample': int(onset),
            'onset_time': float(onset / sr),
            'segment': segment,
            'peak': float(peak),
            'snr': float(snr),
        })

    return isolated


def analyze_spectral_envelope(segment, sr, freq):
    """Analyze harmonic content of a note segment."""
    # Use the first 100ms (strongest tonal content)
    analysis_len = min(int(sr * 0.1), len(segment))
    s = segment[:analysis_len]

    # Compute spectrum
    n_fft = 4096
    S = np.abs(np.fft.rfft(s * np.hanning(len(s)), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)

    # Measure energy at each harmonic
    harmonics = []
    fundamental_energy = 0
    for h in range(1, 16):
        target_freq = freq * h
        if target_freq > sr / 2:
            break

        # Find peak near target frequency (±5% tolerance)
        tol = target_freq * 0.05
        mask = (freqs > target_freq - tol) & (freqs < target_freq + tol)
        if not np.any(mask):
            continue

        energy = float(np.max(S[mask]))
        if h == 1:
            fundamental_energy = energy

        harmonics.append({
            'harmonic': h,
            'target_freq': float(target_freq),
            'energy': energy,
            'db': float(20 * np.log10(energy / max(fundamental_energy, 1e-10))),
        })

    return harmonics


def analyze_envelope(segment, sr):
    """Analyze amplitude envelope: attack time, decay time."""
    # Compute amplitude envelope
    envelope = np.abs(segment)
    # Smooth with 2ms window
    win = max(1, int(sr * 0.002))
    envelope = np.convolve(envelope, np.ones(win) / win, mode='same')

    peak_idx = np.argmax(envelope)
    peak_val = envelope[peak_idx]

    if peak_val < 1e-10:
        return {'attack_ms': 0, 'decay_time': 0}

    # Attack time: time from onset to peak
    attack_ms = peak_idx / sr * 1000

    # Decay time: time from peak to -20dB (10% of peak)
    threshold = peak_val * 0.1
    below = np.where(envelope[peak_idx:] < threshold)[0]
    if len(below) > 0:
        decay_time = below[0] / sr
    else:
        decay_time = (len(segment) - peak_idx) / sr

    # Also find -6dB point for shorter decay estimate
    threshold_6db = peak_val * 0.5
    below_6db = np.where(envelope[peak_idx:] < threshold_6db)[0]
    decay_6db = below_6db[0] / sr if len(below_6db) > 0 else decay_time

    return {
        'attack_ms': float(attack_ms),
        'decay_time': float(decay_time),
        'decay_6db': float(decay_6db),
        'peak_amplitude': float(peak_val),
    }


def analyze_mfcc_profile(segment, sr):
    """Compute MFCC profile of a note segment."""
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
    return {
        'mean': [float(x) for x in np.mean(mfcc, axis=1)],
        'std': [float(x) for x in np.std(mfcc, axis=1)],
    }


def map_freq_to_row(freq, freqs=DEFAULT_FREQS, tolerance_cents=150):
    """Map detected frequency to nearest row."""
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
        return -1, best_dist
    return best_row, best_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', default=None)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref_path = args.ref or os.path.join(root, 'analysis', 'reference.wav')

    if not os.path.exists(ref_path):
        print(f"ERROR: {ref_path} not found")
        sys.exit(1)

    print(f"Loading: {ref_path}")
    y, sr = librosa.load(ref_path, sr=SR, mono=True)
    print(f"  Duration: {len(y)/sr:.1f}s")

    print("\n[1] Finding isolated notes...")
    notes = find_isolated_notes(y, sr, min_silence_ms=50)
    print(f"  Found {len(notes)} isolated notes")

    print("\n[2] Detecting pitches and analyzing...")
    results_by_row = {i: [] for i in range(8)}

    for note in notes:
        segment = note['segment']
        # Detect pitch
        try:
            f0, _, vp = librosa.pyin(segment[:int(sr * 0.1)], fmin=80, fmax=1500, sr=sr)
            f0_valid = f0[~np.isnan(f0)]
            if len(f0_valid) == 0:
                continue
            freq = float(np.median(f0_valid))
        except Exception:
            continue

        row, dist = map_freq_to_row(freq)
        if row < 0:
            continue

        # Analyze this note
        harmonics = analyze_spectral_envelope(segment, sr, DEFAULT_FREQS[row])
        env = analyze_envelope(segment, sr)
        mfcc = analyze_mfcc_profile(segment, sr)

        results_by_row[row].append({
            'detected_freq': freq,
            'row': row,
            'expected_freq': DEFAULT_FREQS[row],
            'onset_time': note['onset_time'],
            'harmonics': harmonics,
            'envelope': env,
            'mfcc': mfcc,
        })

    # Aggregate per row
    print("\n[3] Aggregating results per row...")
    aggregated = {}
    for row in range(8):
        notes_for_row = results_by_row[row]
        if not notes_for_row:
            print(f"  Row {row} ({DEFAULT_FREQS[row]} Hz): no notes detected")
            continue

        print(f"  Row {row} ({DEFAULT_FREQS[row]} Hz): {len(notes_for_row)} notes")

        # Average harmonic profile
        max_harmonics = max(len(n['harmonics']) for n in notes_for_row)
        avg_harmonics = []
        for h in range(max_harmonics):
            dbs = [n['harmonics'][h]['db'] for n in notes_for_row if h < len(n['harmonics'])]
            if dbs:
                avg_harmonics.append(round(float(np.median(dbs)), 1))

        # Average envelope
        attack_ms = np.median([n['envelope']['attack_ms'] for n in notes_for_row])
        decay_time = np.median([n['envelope']['decay_time'] for n in notes_for_row])

        # Average MFCC
        mfcc_means = np.mean([n['mfcc']['mean'] for n in notes_for_row], axis=0)

        aggregated[row] = {
            'freq': DEFAULT_FREQS[row],
            'n_notes': len(notes_for_row),
            'harmonic_db': avg_harmonics,
            'attack_ms': round(float(attack_ms), 1),
            'decay_time': round(float(decay_time), 3),
            'mfcc_mean': [round(float(x), 2) for x in mfcc_means],
        }

        if avg_harmonics:
            print(f"    Harmonics (dB): {avg_harmonics[:8]}")
        print(f"    Attack: {attack_ms:.1f}ms, Decay: {decay_time:.3f}s")

    # Compute global average
    all_harmonic_profiles = [a['harmonic_db'] for a in aggregated.values() if a['harmonic_db']]
    if all_harmonic_profiles:
        max_len = max(len(h) for h in all_harmonic_profiles)
        global_harmonics = []
        for i in range(max_len):
            vals = [h[i] for h in all_harmonic_profiles if i < len(h)]
            global_harmonics.append(round(float(np.median(vals)), 1))

        all_attacks = [a['attack_ms'] for a in aggregated.values()]
        all_decays = [a['decay_time'] for a in aggregated.values()]

        print(f"\n  GLOBAL AVERAGE:")
        print(f"    Harmonics (dB): {global_harmonics[:10]}")
        print(f"    Attack: {np.median(all_attacks):.1f}ms")
        print(f"    Decay: {np.median(all_decays):.3f}s")

        # Format as Python constants
        print(f"\n  Suggested DEFAULT_ENVELOPE:")
        print(f"    'attack_ms': {round(float(np.median(all_attacks)), 1)},")
        print(f"    'decay_time': {round(float(np.median(all_decays)), 3)},")
        print(f"    'harmonic_amplitudes_db': {global_harmonics[:10]},")

    # Also compute the MFCC profile of the reference globally
    print("\n[4] Global reference MFCC profile...")
    ref_mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    ref_mfcc_mean = np.mean(ref_mfcc, axis=1)
    print(f"  Mean MFCCs: {[round(float(x), 2) for x in ref_mfcc_mean]}")

    # Save results
    process_dir = os.path.join(root, 'analysis', 'process')
    os.makedirs(process_dir, exist_ok=True)
    out = {
        'per_row': {str(k): v for k, v in aggregated.items()},
        'global_harmonics_db': global_harmonics[:10] if all_harmonic_profiles else [],
        'global_attack_ms': round(float(np.median(all_attacks)), 1) if all_harmonic_profiles else 1.5,
        'global_decay_time': round(float(np.median(all_decays)), 3) if all_harmonic_profiles else 0.45,
        'reference_mfcc_mean': [round(float(x), 2) for x in ref_mfcc_mean],
    }
    out_path = os.path.join(process_dir, 'tone_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
