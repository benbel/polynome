#!/usr/bin/env python3
"""Prepare reference audio: trim silence, remove artifacts, normalize.

Usage:
    python build/prepare_reference.py [--input analysis/reference_raw.wav]
"""

import argparse, json, os, sys
import numpy as np
import librosa
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

SR = 44100


def highpass(sig, freq, sr=SR, order=4):
    sos = butter(order, max(freq, 20), btype='high', fs=sr, output='sos')
    return sosfilt(sos, sig)


def lowpass(sig, freq, sr=SR, order=4):
    sos = butter(order, min(freq, sr / 2 - 1), btype='low', fs=sr, output='sos')
    return sosfilt(sos, sig)


def estimate_noise_floor(y, sr, duration=0.5):
    """Estimate noise floor from the quietest segment."""
    frame_len = int(sr * 0.05)
    hop = frame_len // 2
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]

    # Find the quietest contiguous segment of `duration` seconds
    n_frames = int(duration * sr / hop)
    if n_frames >= len(rms):
        return float(np.mean(rms))

    # Sliding window average
    cumsum = np.cumsum(rms)
    window_sums = cumsum[n_frames:] - cumsum[:-n_frames]
    quietest_start = int(np.argmin(window_sums))

    noise_rms = float(np.mean(rms[quietest_start:quietest_start + n_frames]))
    return noise_rms


def spectral_gate(y, sr, noise_rms, margin_db=6):
    """Simple spectral gating: zero out STFT bins below noise threshold."""
    n_fft = 2048
    hop = 512
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag = np.abs(S)
    phase = np.angle(S)

    threshold = noise_rms * (10 ** (margin_db / 20))
    # Gate: reduce bins below threshold
    mask = mag > threshold
    # Soft gate: attenuate rather than zero
    gain = np.where(mask, 1.0, mag / (threshold + 1e-10) * 0.1)
    S_gated = (mag * gain) * np.exp(1j * phase)

    return librosa.istft(S_gated, hop_length=hop, length=len(y))


def main():
    parser = argparse.ArgumentParser(description='Prepare reference audio')
    parser.add_argument('--input', default=None)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    analysis_dir = os.path.join(root, 'analysis')
    process_dir = os.path.join(analysis_dir, 'process')
    os.makedirs(process_dir, exist_ok=True)

    in_path = args.input or os.path.join(analysis_dir, 'reference_raw.wav')
    out_path = args.output or os.path.join(analysis_dir, 'reference.wav')

    if not os.path.exists(in_path):
        print(f'ERROR: input not found: {in_path}')
        sys.exit(1)

    # Load
    print(f'Loading: {in_path}')
    y, sr = librosa.load(in_path, sr=SR, mono=True)
    raw_duration = len(y) / sr
    print(f'  Duration: {raw_duration:.1f}s, samples: {len(y)}')

    diagnostics = {
        'raw_duration_s': round(raw_duration, 2),
        'sample_rate': sr,
    }

    # Step 1: Trim silence
    print('Trimming silence...')
    y_trimmed, trim_indices = librosa.effects.trim(y, top_db=30)
    trim_start_s = trim_indices[0] / sr
    trim_end_s = trim_indices[1] / sr
    print(f'  Trimmed: {trim_start_s:.2f}s - {trim_end_s:.2f}s '
          f'({len(y_trimmed) / sr:.1f}s)')
    diagnostics['trim_start_s'] = round(trim_start_s, 3)
    diagnostics['trim_end_s'] = round(trim_end_s, 3)
    diagnostics['trimmed_duration_s'] = round(len(y_trimmed) / sr, 2)

    # Validate: first onset should be near start
    onsets = librosa.onset.onset_detect(y=y_trimmed, sr=sr, units='time')
    if len(onsets) > 0:
        print(f'  First onset at: {onsets[0]:.3f}s (should be < 0.5s)')
        diagnostics['first_onset_s'] = round(float(onsets[0]), 3)

    y = y_trimmed

    # Step 2: High-pass filter (remove rumble/hum)
    print('Applying high-pass filter at 70 Hz...')
    y = highpass(y, 70, sr)

    # Step 3: Low-pass filter (remove HF noise above musical content)
    print('Applying low-pass filter at 12 kHz...')
    y = lowpass(y, 12000, sr)

    # Step 4: Estimate noise floor and apply spectral gating
    print('Estimating noise floor...')
    noise_rms = estimate_noise_floor(y, sr, duration=0.5)
    print(f'  Noise floor RMS: {noise_rms:.6f}')
    diagnostics['noise_floor_rms'] = round(float(noise_rms), 6)

    print('Applying spectral gating (aggressive)...')
    y = spectral_gate(y, sr, noise_rms, margin_db=12)

    # Step 5: Detect mid-performance silence gaps
    frame_len = int(sr * 0.1)
    hop = frame_len // 2
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    silence_threshold = np.median(rms) * 0.05
    silent_frames = rms < silence_threshold
    # Find contiguous silence > 1s
    gap_starts = []
    in_gap = False
    gap_start = 0
    for i in range(len(silent_frames)):
        if silent_frames[i] and not in_gap:
            in_gap = True
            gap_start = i
        elif not silent_frames[i] and in_gap:
            gap_dur = (i - gap_start) * hop / sr
            if gap_dur > 1.0:
                gap_starts.append({
                    'start_s': round(float(rms_times[gap_start]), 2),
                    'end_s': round(float(rms_times[i]), 2),
                    'duration_s': round(gap_dur, 2),
                })
            in_gap = False
    if gap_starts:
        print(f'  Found {len(gap_starts)} silence gaps > 1s:')
        for g in gap_starts:
            print(f"    {g['start_s']:.1f}s - {g['end_s']:.1f}s ({g['duration_s']:.1f}s)")
    diagnostics['silence_gaps'] = gap_starts

    # Step 6: RMS normalize
    print('Normalizing...')
    current_rms = np.sqrt(np.mean(y ** 2))
    if current_rms > 0:
        target_rms = 0.30
        gain = target_rms / current_rms
        y *= gain
    y = np.clip(y, -0.95, 0.95)

    diagnostics['output_rms'] = round(float(np.sqrt(np.mean(y ** 2))), 4)
    diagnostics['output_peak'] = round(float(np.max(np.abs(y))), 4)
    diagnostics['output_duration_s'] = round(len(y) / sr, 2)

    # Save RMS envelope for analysis
    rms_out = librosa.feature.rms(y=y, frame_length=int(sr * 0.05),
                                   hop_length=int(sr * 0.025))[0]
    diagnostics['rms_envelope_mean'] = round(float(np.mean(rms_out)), 4)
    diagnostics['rms_envelope_std'] = round(float(np.std(rms_out)), 4)

    # Write output
    sig16 = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    wavfile.write(out_path, SR, sig16)
    print(f'Written: {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)')

    # Write diagnostics
    diag_path = os.path.join(process_dir, 'reference_diagnostics.json')
    with open(diag_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    print(f'Diagnostics: {diag_path}')

    print('\nDone!')


if __name__ == '__main__':
    main()
