#!/usr/bin/env python3
"""Compare reference audio (from video) with rendered audio.

Computes decomposed distance metrics and outputs a JSON report + appends
to a CSV history log.

Usage:
    python build/compare_audio.py [--ref analysis/reference.wav] [--rendered analysis/rendered.wav]
"""

import argparse, csv, json, os, sys
import numpy as np
import librosa


def load_audio(path, sr=22050):
    """Load audio file as mono at target sample rate."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y, sr


def spectral_centroid_distance(y_ref, y_ren, sr):
    """Mean absolute difference of frame-wise spectral centroids (Hz)."""
    sc_ref = librosa.feature.spectral_centroid(y=y_ref, sr=sr)[0]
    sc_ren = librosa.feature.spectral_centroid(y=y_ren, sr=sr)[0]
    # Align lengths
    n = min(len(sc_ref), len(sc_ren))
    diff = np.abs(sc_ref[:n] - sc_ren[:n])
    # Normalize by Nyquist
    return float(np.mean(diff) / (sr / 2))


def mfcc_distance(y_ref, y_ren, sr):
    """Euclidean distance between mean MFCC vectors (13 coefficients)."""
    mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr, n_mfcc=13)
    mfcc_ren = librosa.feature.mfcc(y=y_ren, sr=sr, n_mfcc=13)
    mean_ref = np.mean(mfcc_ref, axis=1)
    mean_ren = np.mean(mfcc_ren, axis=1)
    dist = np.linalg.norm(mean_ref - mean_ren)
    # Normalize: typical MFCC distance range is 0–200
    return float(min(dist / 200, 1.0))


def onset_density_ratio(y_ref, y_ren, sr):
    """Ratio of onset counts per second (0 = identical, 1 = very different)."""
    onsets_ref = librosa.onset.onset_detect(y=y_ref, sr=sr)
    onsets_ren = librosa.onset.onset_detect(y=y_ren, sr=sr)
    dur_ref = len(y_ref) / sr
    dur_ren = len(y_ren) / sr
    rate_ref = len(onsets_ref) / max(dur_ref, 0.1)
    rate_ren = len(onsets_ren) / max(dur_ren, 0.1)
    if rate_ref == 0 and rate_ren == 0:
        return 0.0
    ratio = abs(rate_ref - rate_ren) / max(rate_ref, rate_ren)
    return float(ratio)


def ioi_histogram_distance(y_ref, y_ren, sr):
    """Earth-mover (Wasserstein) distance between inter-onset-interval histograms."""
    from scipy.stats import wasserstein_distance

    onsets_ref = librosa.onset.onset_detect(y=y_ref, sr=sr, units='time')
    onsets_ren = librosa.onset.onset_detect(y=y_ren, sr=sr, units='time')

    if len(onsets_ref) < 3 or len(onsets_ren) < 3:
        return 1.0

    ioi_ref = np.diff(onsets_ref)
    ioi_ren = np.diff(onsets_ren)

    # Compute Wasserstein distance, normalize by max IOI
    max_ioi = max(np.max(ioi_ref), np.max(ioi_ren), 0.01)
    dist = wasserstein_distance(ioi_ref, ioi_ren)
    return float(min(dist / max_ioi, 1.0))


def rms_envelope_correlation(y_ref, y_ren, sr):
    """1 - cross-correlation of smoothed RMS envelopes (0 = identical, 1 = uncorrelated)."""
    frame_length = int(sr * 0.05)  # 50ms frames
    hop_length = frame_length // 2

    rms_ref = librosa.feature.rms(y=y_ref, frame_length=frame_length, hop_length=hop_length)[0]
    rms_ren = librosa.feature.rms(y=y_ren, frame_length=frame_length, hop_length=hop_length)[0]

    # Align lengths
    n = min(len(rms_ref), len(rms_ren))
    rms_ref = rms_ref[:n]
    rms_ren = rms_ren[:n]

    if np.std(rms_ref) < 1e-8 or np.std(rms_ren) < 1e-8:
        return 1.0

    corr = np.corrcoef(rms_ref, rms_ren)[0, 1]
    return float(max(0, 1 - corr))


def pitch_class_distance(y_ref, y_ren, sr):
    """Cosine distance between chromagram histograms."""
    chroma_ref = librosa.feature.chroma_stft(y=y_ref, sr=sr)
    chroma_ren = librosa.feature.chroma_stft(y=y_ren, sr=sr)
    hist_ref = np.mean(chroma_ref, axis=1)
    hist_ren = np.mean(chroma_ren, axis=1)

    dot = np.dot(hist_ref, hist_ren)
    norm = np.linalg.norm(hist_ref) * np.linalg.norm(hist_ren)
    if norm < 1e-8:
        return 1.0
    cosine_sim = dot / norm
    return float(max(0, 1 - cosine_sim))


def spectral_rolloff_distance(y_ref, y_ren, sr):
    """Mean absolute difference of spectral rolloff frequencies."""
    ro_ref = librosa.feature.spectral_rolloff(y=y_ref, sr=sr)[0]
    ro_ren = librosa.feature.spectral_rolloff(y=y_ren, sr=sr)[0]
    n = min(len(ro_ref), len(ro_ren))
    diff = np.abs(ro_ref[:n] - ro_ren[:n])
    return float(np.mean(diff) / (sr / 2))


def diagnose(metrics):
    """Generate actionable diagnosis from metrics."""
    tips = []
    if metrics['spectral_centroid'] > 0.15:
        tips.append('spectral_centroid HIGH → reduce upper harmonics or add lowpass')
    if metrics['spectral_rolloff'] > 0.15:
        tips.append('spectral_rolloff HIGH → reduce saturation/comb filter, fewer harmonics')
    if metrics['mfcc'] > 0.4:
        tips.append('mfcc HIGH → envelope or harmonic structure mismatch')
    if metrics['onset_density'] > 0.3:
        tips.append('onset_density HIGH → adjust step_ms or COL_SEQS')
    if metrics['ioi_histogram'] > 0.3:
        tips.append('ioi_histogram HIGH → rhythmic structure mismatch, check COL_SEQS')
    if metrics['rms_correlation'] > 0.5:
        tips.append('rms_correlation HIGH → dynamic shape off, check pattern order/velocity')
    if metrics['pitch_class'] > 0.2:
        tips.append('pitch_class HIGH → frequency mismatch, check mainFreqs')
    if not tips:
        tips.append('All metrics within acceptable range')
    return tips


def compute_distance(ref_path, rendered_path):
    """Compute all sub-metrics and a weighted composite score."""
    print(f'Loading reference: {ref_path}')
    y_ref, sr = load_audio(ref_path)
    print(f'Loading rendered:  {rendered_path}')
    y_ren, sr = load_audio(rendered_path)

    print(f'  Reference:  {len(y_ref) / sr:.1f}s')
    print(f'  Rendered:   {len(y_ren) / sr:.1f}s')

    # Trim both to the shorter duration for fair comparison
    min_len = min(len(y_ref), len(y_ren))
    y_ref = y_ref[:min_len]
    y_ren = y_ren[:min_len]

    print('Computing metrics...')
    metrics = {}

    print('  spectral_centroid...')
    metrics['spectral_centroid'] = spectral_centroid_distance(y_ref, y_ren, sr)

    print('  mfcc...')
    metrics['mfcc'] = mfcc_distance(y_ref, y_ren, sr)

    print('  onset_density...')
    metrics['onset_density'] = onset_density_ratio(y_ref, y_ren, sr)

    print('  ioi_histogram...')
    metrics['ioi_histogram'] = ioi_histogram_distance(y_ref, y_ren, sr)

    print('  rms_correlation...')
    metrics['rms_correlation'] = rms_envelope_correlation(y_ref, y_ren, sr)

    print('  pitch_class...')
    metrics['pitch_class'] = pitch_class_distance(y_ref, y_ren, sr)

    print('  spectral_rolloff...')
    metrics['spectral_rolloff'] = spectral_rolloff_distance(y_ref, y_ren, sr)

    # Weighted composite — timbre metrics weighted higher
    weights = {
        'spectral_centroid': 0.15,
        'mfcc': 0.25,
        'onset_density': 0.10,
        'ioi_histogram': 0.10,
        'rms_correlation': 0.15,
        'pitch_class': 0.10,
        'spectral_rolloff': 0.15,
    }
    composite = sum(metrics[k] * weights[k] for k in weights)
    metrics['composite'] = round(composite, 6)

    # Round sub-metrics
    for k in metrics:
        metrics[k] = round(metrics[k], 6)

    metrics['diagnosis'] = diagnose(metrics)
    return metrics


def append_history(history_path, iteration, composite, description):
    """Append one row to the CSV history log."""
    exists = os.path.exists(history_path)
    with open(history_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(['iteration', 'composite_distance', 'description'])
        writer.writerow([iteration, composite, description])


def main():
    parser = argparse.ArgumentParser(description='Compare reference vs rendered audio')
    parser.add_argument('--ref', default=None, help='Path to reference WAV')
    parser.add_argument('--rendered', default=None, help='Path to rendered WAV')
    parser.add_argument('--iteration', type=int, default=0, help='Iteration number')
    parser.add_argument('--description', default='baseline', help='Description of this iteration')
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    analysis_dir = os.path.join(root, 'analysis')

    ref_path = args.ref or os.path.join(analysis_dir, 'reference.wav')
    rendered_path = args.rendered or os.path.join(analysis_dir, 'rendered.wav')

    if not os.path.exists(ref_path):
        print(f'ERROR: reference file not found: {ref_path}')
        sys.exit(1)
    if not os.path.exists(rendered_path):
        print(f'ERROR: rendered file not found: {rendered_path}')
        sys.exit(1)

    metrics = compute_distance(ref_path, rendered_path)

    # Print report
    print('\n' + '=' * 60)
    print('DISTANCE REPORT')
    print('=' * 60)
    for k, v in metrics.items():
        if k == 'diagnosis':
            continue
        print(f'  {k:25s}: {v:.6f}')
    print('-' * 60)
    print('DIAGNOSIS:')
    for tip in metrics['diagnosis']:
        print(f'  → {tip}')
    print('=' * 60)

    # Write JSON report
    report_path = os.path.join(analysis_dir, 'distance.json')
    with open(report_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\nReport written: {report_path}')

    # Append to CSV history
    history_path = os.path.join(analysis_dir, 'history.csv')
    append_history(history_path, args.iteration, metrics['composite'], args.description)
    print(f'History appended: {history_path}')


if __name__ == '__main__':
    main()
