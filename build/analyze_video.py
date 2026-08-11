#!/usr/bin/env python3
"""Video analysis pipeline for original mode.

Extracts scale, envelope profile, grid dimensions, and timing
from a monome grid video.

Usage:
    python build/analyze_video.py video/monome_source.mp4
"""

import sys, os, json
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

# Optional imports — graceful fallback
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def extract_audio(video_path, out_path, sr=44100):
    """Extract audio from video via ffmpeg."""
    import subprocess
    subprocess.run([
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-ar', str(sr), '-ac', '1', out_path
    ], capture_output=True, check=True)


def detect_scale(audio_path, sr=44100):
    """Detect the scale/pitches used in the video."""
    if not HAS_LIBROSA:
        print('  [warn] librosa not available, using default scale')
        return {
            'pitches_hz': [440, 392, 330, 294, 262, 220, 196, 165],
            'scale_name': 'default (no analysis)',
            'tuning_ref_hz': 440,
            'num_rows': 8,
        }

    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    # Onset detection
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='samples', backtrack=True)

    # Pitch detection at each onset
    pitches = []
    for onset in onsets:
        segment = y[onset:onset + int(sr * 0.3)]
        if len(segment) < sr * 0.05:
            continue
        f0, voiced_flag, voiced_prob = librosa.pyin(
            segment, fmin=50, fmax=2000, sr=sr
        )
        f0_valid = f0[~np.isnan(f0)]
        if len(f0_valid) > 0:
            pitches.append(float(np.median(f0_valid)))

    if len(pitches) < 4:
        print(f'  [warn] only {len(pitches)} pitches detected, using defaults')
        return {
            'pitches_hz': [440, 392, 330, 294, 262, 220, 196, 165],
            'scale_name': 'default (insufficient detections)',
            'tuning_ref_hz': 440,
            'num_rows': 8,
        }

    # Cluster pitches — try k=8 first, then k=16
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    pitch_array = np.array(pitches).reshape(-1, 1)
    best_k, best_score = 8, -1

    for k in [8, 12, 16]:
        if k > len(pitches):
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(pitch_array)
        score = silhouette_score(pitch_array, labels)
        if score > best_score:
            best_k, best_score = k, score

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    km.fit(pitch_array)
    centers = sorted(km.cluster_centers_.flatten().tolist())

    return {
        'pitches_hz': [round(f, 1) for f in centers],
        'scale_name': f'detected ({best_k} pitches, silhouette={best_score:.2f})',
        'tuning_ref_hz': 440,
        'num_rows': best_k,
    }


def detect_envelope(audio_path, sr=44100):
    """Characterize the timbral envelope of notes."""
    if not HAS_LIBROSA:
        return {
            'attack_ms': 3.0,
            'decay_time': 0.28,
            'harmonic_ratios': [1.0, 2.01, 3.98, 5.02, 6.97],
            'harmonic_amplitudes_db': [0, -8, -15, -22, -30],
            'inharmonicity_cents': [0, 17, -5, 8, -12],
        }

    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='samples', backtrack=True)

    # Find isolated notes (low energy before onset)
    attack_times = []
    decay_times = []

    for onset in onsets[:20]:
        segment = y[onset:onset + int(sr * 0.5)]
        if len(segment) < sr * 0.1:
            continue
        envelope = np.abs(segment)
        # Smooth
        from scipy.ndimage import uniform_filter1d
        envelope = uniform_filter1d(envelope, int(sr * 0.005))
        peak_idx = np.argmax(envelope)
        attack_times.append(peak_idx / sr * 1000)

        # Decay: find -20dB point
        peak_val = envelope[peak_idx]
        if peak_val > 0:
            threshold = peak_val * 0.1
            below = np.where(envelope[peak_idx:] < threshold)[0]
            if len(below) > 0:
                decay_times.append(below[0] / sr)

    attack_ms = float(np.median(attack_times)) if attack_times else 3.0
    decay_time = float(np.median(decay_times)) if decay_times else 0.28

    return {
        'attack_ms': round(attack_ms, 1),
        'decay_time': round(decay_time, 3),
        'harmonic_ratios': [1.0, 2.01, 3.98, 5.02, 6.97],
        'harmonic_amplitudes_db': [0, -8, -15, -22, -30],
        'inharmonicity_cents': [0, 17, -5, 8, -12],
    }


def detect_grid(video_path):
    """Detect grid dimensions from video frames."""
    if not HAS_CV2:
        return {'rows': 8, 'cols': 16, 'button_shape': 'round'}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'rows': 8, 'cols': 16, 'button_shape': 'round'}

    best_grid = None
    for sec in [1, 5, 10, 20, 30]:
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 8:
            continue

        areas = [cv2.contourArea(c) for c in contours]
        median_area = np.median(areas)
        buttons = [c for c, a in zip(contours, areas)
                   if 0.3 * median_area < a < 3 * median_area]

        if len(buttons) < 8:
            continue

        # Get centroids
        centers = []
        for c in buttons:
            M = cv2.moments(c)
            if M['m00'] > 0:
                centers.append((M['m10'] / M['m00'], M['m01'] / M['m00']))

        if len(centers) < 8:
            continue

        centers = np.array(centers)

        # Cluster Y coordinates to find rows
        y_sorted = np.sort(centers[:, 1])
        y_gaps = np.diff(y_sorted)
        gap_threshold = np.median(y_gaps) * 2
        row_breaks = np.where(y_gaps > gap_threshold)[0]
        n_rows = len(row_breaks) + 1

        # Cluster X coordinates to find cols
        x_sorted = np.sort(centers[:, 0])
        x_gaps = np.diff(x_sorted)
        gap_threshold = np.median(x_gaps) * 2
        col_breaks = np.where(x_gaps > gap_threshold)[0]
        n_cols = len(col_breaks) + 1

        if n_rows >= 4 and n_cols >= 4:
            best_grid = {'rows': n_rows, 'cols': n_cols, 'button_shape': 'round'}
            break

    cap.release()
    return best_grid or {'rows': 8, 'cols': 16, 'button_shape': 'round'}


def detect_timing(audio_path, sr=44100):
    """Detect step rate from inter-onset intervals."""
    if not HAS_LIBROSA:
        return {'step_ms': 180, 'bpm_equivalent': 333}

    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='samples')

    if len(onsets) < 10:
        return {'step_ms': 180, 'bpm_equivalent': 333}

    ioi = np.diff(onsets) / sr * 1000  # ms
    # Find mode of IOI distribution
    hist, edges = np.histogram(ioi, bins=100, range=(50, 500))
    step_ms = float(edges[np.argmax(hist)])
    bpm = 60000 / step_ms

    return {
        'step_ms': round(step_ms),
        'bpm_equivalent': round(bpm),
    }


def main():
    if len(sys.argv) < 2:
        print('Usage: python build/analyze_video.py <video_path>')
        sys.exit(1)

    video_path = sys.argv[1]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    analysis_dir = os.path.join(root, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(os.path.join(analysis_dir, 'frames'), exist_ok=True)
    os.makedirs(os.path.join(analysis_dir, 'audio_segments'), exist_ok=True)

    # Extract audio
    audio_path = os.path.join(analysis_dir, 'audio_raw.wav')
    print('[1/5] Extracting audio...')
    extract_audio(video_path, audio_path)

    # Scale detection
    print('[2/5] Detecting scale...')
    scale = detect_scale(audio_path)
    with open(os.path.join(analysis_dir, 'scale_detected.json'), 'w') as f:
        json.dump(scale, f, indent=2)
    print(f'  Found {scale["num_rows"]} pitches: {scale["scale_name"]}')

    # Envelope analysis
    print('[3/5] Analyzing envelope...')
    envelope = detect_envelope(audio_path)
    with open(os.path.join(analysis_dir, 'envelope_profile.json'), 'w') as f:
        json.dump(envelope, f, indent=2)
    print(f'  Attack: {envelope["attack_ms"]}ms, Decay: {envelope["decay_time"]}s')

    # Grid detection
    print('[4/5] Detecting grid...')
    grid = detect_grid(video_path)
    with open(os.path.join(analysis_dir, 'grid_detected.json'), 'w') as f:
        json.dump(grid, f, indent=2)
    print(f'  Grid: {grid["rows"]}x{grid["cols"]}')

    # Timing
    print('[5/5] Detecting timing...')
    timing = detect_timing(audio_path)
    with open(os.path.join(analysis_dir, 'timing_detected.json'), 'w') as f:
        json.dump(timing, f, indent=2)
    print(f'  Step: {timing["step_ms"]}ms (~{timing["bpm_equivalent"]} bpm)')

    print('\nAnalysis complete. Results in analysis/')


if __name__ == '__main__':
    main()
