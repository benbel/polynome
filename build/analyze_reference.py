#!/usr/bin/env python3
"""Deep statistical analysis of cleaned reference audio.

Extracts: onset timing, per-segment RMS, pitch content, note density.

Usage:
    python build/analyze_reference.py [--ref analysis/reference.wav]
"""

import argparse, json, os, sys
import numpy as np
import librosa

SR = 44100


def analyze_onsets(y, sr):
    """Onset timing statistics."""
    print('  Detecting onsets...')
    onsets_time = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    onsets_samples = librosa.onset.onset_detect(y=y, sr=sr, units='samples')

    if len(onsets_time) < 3:
        return {'error': 'too few onsets detected', 'count': len(onsets_time)}

    ioi = np.diff(onsets_time) * 1000  # ms
    print(f'  {len(onsets_time)} onsets detected')
    print(f'  IOI stats: mean={np.mean(ioi):.1f}ms, median={np.median(ioi):.1f}ms, '
          f'std={np.std(ioi):.1f}ms, min={np.min(ioi):.1f}ms, max={np.max(ioi):.1f}ms')

    # Histogram of IOIs to find modes
    hist, edges = np.histogram(ioi, bins=100, range=(20, 500))
    peak_bins = np.argsort(hist)[-5:]  # Top 5 bins
    peak_iois = [(edges[b] + edges[b + 1]) / 2 for b in sorted(peak_bins)]
    modal_ioi = peak_iois[np.argmax([hist[b] for b in sorted(peak_bins)])]

    print(f'  Modal IOI: {modal_ioi:.1f}ms')
    print(f'  Top IOI peaks: {[round(p, 1) for p in peak_iois]}')

    # Check for tempo variation: split into 10s windows
    window_dur = 10.0
    n_windows = int(np.ceil(onsets_time[-1] / window_dur))
    local_iois = []
    for w in range(n_windows):
        t_start = w * window_dur
        t_end = (w + 1) * window_dur
        mask = (onsets_time >= t_start) & (onsets_time < t_end)
        window_onsets = onsets_time[mask]
        if len(window_onsets) > 2:
            w_ioi = np.diff(window_onsets) * 1000
            local_iois.append({
                'window_s': round(t_start, 1),
                'onset_count': len(window_onsets),
                'mean_ioi_ms': round(float(np.mean(w_ioi)), 1),
                'median_ioi_ms': round(float(np.median(w_ioi)), 1),
                'density_per_s': round(len(window_onsets) / window_dur, 1),
            })

    return {
        'onset_count': len(onsets_time),
        'duration_s': round(float(onsets_time[-1]), 2),
        'ioi_mean_ms': round(float(np.mean(ioi)), 1),
        'ioi_median_ms': round(float(np.median(ioi)), 1),
        'ioi_std_ms': round(float(np.std(ioi)), 1),
        'ioi_min_ms': round(float(np.min(ioi)), 1),
        'ioi_max_ms': round(float(np.max(ioi)), 1),
        'modal_ioi_ms': round(modal_ioi, 1),
        'top_ioi_peaks_ms': [round(p, 1) for p in peak_iois],
        'ioi_histogram_top10': [
            {'center_ms': round((edges[b] + edges[b + 1]) / 2, 1), 'count': int(hist[b])}
            for b in np.argsort(hist)[-10:][::-1]
        ],
        'local_windows': local_iois,
    }


def analyze_rms_segments(y, sr, step_ms=74, steps_per_pattern=64):
    """Per-pattern-segment RMS analysis."""
    print(f'  Computing per-segment RMS (step_ms={step_ms}, steps/pattern={steps_per_pattern})...')

    pattern_dur_s = step_ms / 1000 * steps_per_pattern
    pattern_dur_samples = int(pattern_dur_s * sr)
    total_dur = len(y) / sr
    n_segments = int(total_dur / pattern_dur_s)

    segments = []
    for i in range(n_segments):
        start = i * pattern_dur_samples
        end = min(start + pattern_dur_samples, len(y))
        seg = y[start:end]
        seg_rms = float(np.sqrt(np.mean(seg ** 2)))
        seg_peak = float(np.max(np.abs(seg)))

        # Count onsets in this segment
        onset_frames = librosa.onset.onset_detect(y=seg, sr=sr)
        segments.append({
            'pattern_idx': i,
            'start_s': round(i * pattern_dur_s, 2),
            'rms': round(seg_rms, 4),
            'peak': round(seg_peak, 4),
            'onset_count': len(onset_frames),
        })

    rms_values = [s['rms'] for s in segments]
    print(f'  {n_segments} segments, RMS range: {min(rms_values):.4f} - {max(rms_values):.4f}')

    # Dynamic arc characterization
    if n_segments >= 4:
        q1 = np.mean(rms_values[:n_segments // 4])
        q4 = np.mean(rms_values[-n_segments // 4:])
        print(f'  First quarter avg RMS: {q1:.4f}, Last quarter: {q4:.4f}')
        print(f'  Dynamic ratio (last/first): {q4 / max(q1, 1e-6):.2f}x')

    return {
        'step_ms': step_ms,
        'steps_per_pattern': steps_per_pattern,
        'pattern_duration_s': round(pattern_dur_s, 3),
        'n_segments': n_segments,
        'segments': segments,
    }


def analyze_pitches(y, sr):
    """Pitch detection and clustering."""
    print('  Running pitch detection (pYIN)...')

    # Detect onsets and analyze pitch at each
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='samples', backtrack=True)
    print(f'  Analyzing pitches at {min(200, len(onsets))} onsets...')

    pitches = []
    for onset in onsets[:200]:  # Cap at 200 for speed
        segment = y[onset:onset + int(sr * 0.15)]
        if len(segment) < sr * 0.03:
            continue
        try:
            f0, voiced_flag, voiced_prob = librosa.pyin(
                segment, fmin=50, fmax=2000, sr=sr
            )
            f0_valid = f0[~np.isnan(f0)]
            if len(f0_valid) > 0:
                median_f0 = float(np.median(f0_valid))
                confidence = float(np.mean(voiced_prob[~np.isnan(f0)]))
                pitches.append({'hz': median_f0, 'confidence': confidence})
        except Exception:
            continue

    if len(pitches) < 4:
        return {'error': f'only {len(pitches)} pitches detected'}

    # Cluster pitches
    from sklearn.cluster import KMeans

    hz_values = np.array([p['hz'] for p in pitches]).reshape(-1, 1)

    best_k, best_score = 8, -1
    for k in [6, 7, 8, 9, 10]:
        if k > len(pitches):
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(hz_values)
        from sklearn.metrics import silhouette_score
        score = silhouette_score(hz_values, labels)
        if score > best_score:
            best_k, best_score = k, score

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(hz_values)
    centers = sorted(km.cluster_centers_.flatten().tolist())

    # Count per cluster
    cluster_counts = {}
    for label in labels:
        c = round(centers[label], 1) if label < len(centers) else 0
        cluster_counts[c] = cluster_counts.get(c, 0) + 1

    # Convert to MIDI note names
    def hz_to_note(hz):
        if hz <= 0:
            return 'N/A'
        midi = 69 + 12 * np.log2(hz / 440)
        names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note = names[int(round(midi)) % 12]
        octave = int(round(midi)) // 12 - 1
        cents_off = round((midi - round(midi)) * 100)
        return f'{note}{octave} ({cents_off:+d}c)'

    print(f'  Found {best_k} pitch clusters (silhouette={best_score:.2f}):')
    detected_freqs = []
    for c in centers:
        note = hz_to_note(c)
        count = cluster_counts.get(round(c, 1), 0)
        print(f'    {c:.1f} Hz = {note} (n={count})')
        detected_freqs.append(round(c, 1))

    # Compare with AMXD and current
    amxd_freqs = [261.6, 246.9, 220.0, 196.0, 174.6, 164.8, 146.8, 130.8]
    current_freqs = [587, 494, 415, 349, 294, 247, 196, 147]

    return {
        'n_pitches_detected': len(pitches),
        'n_clusters': best_k,
        'silhouette_score': round(best_score, 3),
        'cluster_centers_hz': detected_freqs,
        'cluster_notes': [hz_to_note(c) for c in centers],
        'cluster_counts': {str(round(c, 1)): cluster_counts.get(round(c, 1), 0) for c in centers},
        'comparison': {
            'amxd_freqs': amxd_freqs,
            'current_freqs': current_freqs,
            'detected_freqs': detected_freqs,
        }
    }


def analyze_density(y, sr, step_ms=74, steps_per_pattern=64):
    """Note density over time."""
    print('  Computing note density over time...')
    pattern_dur_s = step_ms / 1000 * steps_per_pattern
    total_dur = len(y) / sr

    onsets_time = librosa.onset.onset_detect(y=y, sr=sr, units='time')

    # Density per pattern-length window
    n_windows = int(total_dur / pattern_dur_s)
    density = []
    for i in range(n_windows):
        t_start = i * pattern_dur_s
        t_end = (i + 1) * pattern_dur_s
        count = np.sum((onsets_time >= t_start) & (onsets_time < t_end))
        density.append({
            'segment': i,
            'start_s': round(t_start, 2),
            'onset_count': int(count),
            'density_per_s': round(count / pattern_dur_s, 1),
        })

    densities = [d['onset_count'] for d in density]
    if len(densities) > 4:
        q1 = np.mean(densities[:len(densities) // 4])
        q4 = np.mean(densities[-len(densities) // 4:])
        print(f'  First quarter avg onsets: {q1:.1f}, Last quarter: {q4:.1f}')
        print(f'  Density buildup ratio: {q4 / max(q1, 0.1):.1f}x')

    return {
        'total_onsets': len(onsets_time),
        'n_windows': n_windows,
        'windows': density,
    }


def analyze_spectral_envelope(y, sr):
    """Average spectral envelope."""
    print('  Computing average spectral envelope...')
    S = np.abs(librosa.stft(y, n_fft=4096))
    avg_spectrum = np.mean(S, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

    # Find dominant frequencies
    peaks_idx = np.argsort(avg_spectrum)[-20:]
    dominant = sorted([(float(freqs[i]), float(avg_spectrum[i])) for i in peaks_idx],
                      key=lambda x: -x[1])

    print(f'  Top 5 spectral peaks: {[(round(f, 1), round(a, 4)) for f, a in dominant[:5]]}')

    # Energy in frequency bands
    bands = {
        'sub_bass_20_60': (20, 60),
        'bass_60_250': (60, 250),
        'low_mid_250_500': (250, 500),
        'mid_500_2000': (500, 2000),
        'upper_mid_2000_4000': (2000, 4000),
        'presence_4000_8000': (4000, 8000),
        'brilliance_8000_12000': (8000, 12000),
    }
    band_energy = {}
    total_energy = float(np.sum(avg_spectrum ** 2))
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        energy = float(np.sum(avg_spectrum[mask] ** 2))
        band_energy[name] = round(energy / max(total_energy, 1e-10), 4)

    return {
        'dominant_freqs': [{'hz': round(f, 1), 'magnitude': round(a, 4)} for f, a in dominant[:10]],
        'band_energy_ratios': band_energy,
    }


def main():
    parser = argparse.ArgumentParser(description='Deep analysis of reference audio')
    parser.add_argument('--ref', default=None)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    analysis_dir = os.path.join(root, 'analysis')
    process_dir = os.path.join(analysis_dir, 'process')
    os.makedirs(process_dir, exist_ok=True)

    ref_path = args.ref or os.path.join(analysis_dir, 'reference.wav')
    if not os.path.exists(ref_path):
        print(f'ERROR: {ref_path} not found')
        sys.exit(1)

    print(f'Loading: {ref_path}')
    y, sr = librosa.load(ref_path, sr=SR, mono=True)
    print(f'  Duration: {len(y) / sr:.1f}s')

    results = {}

    print('\n[1/5] Onset Timing Analysis')
    results['onsets'] = analyze_onsets(y, sr)

    print('\n[2/5] Per-Segment RMS Analysis')
    results['rms_segments'] = analyze_rms_segments(y, sr, step_ms=74, steps_per_pattern=64)

    print('\n[3/5] Pitch Detection')
    results['pitches'] = analyze_pitches(y, sr)

    print('\n[4/5] Note Density Over Time')
    results['density'] = analyze_density(y, sr, step_ms=74, steps_per_pattern=64)

    print('\n[5/5] Spectral Envelope')
    results['spectral'] = analyze_spectral_envelope(y, sr)

    out_path = os.path.join(process_dir, 'reference_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults written: {out_path}')


if __name__ == '__main__':
    main()
