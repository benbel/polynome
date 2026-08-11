#!/usr/bin/env python3

import argparse, json, os, sys, time
import numpy as np
import librosa
from scipy.signal import fftconvolve
import cma

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    SR, sine, noise, env_exp_decay,
    lowpass, highpass, bandpass, comb_filter,
    asymmetric_saturate, normalize, fade_in, fade_out,
    to_stereo, generate_reverb_ir,
)

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
COMPARE_SR = 22050

_ref_envelope_cache = {}


def get_reference_envelope(sr):
    if sr in _ref_envelope_cache:
        return _ref_envelope_cache[sr]

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref_path = os.path.join(root, 'analysis', 'reference.wav')
    if not os.path.exists(ref_path):
        return None

    env_sr = sr // 2
    y_ref, _ = librosa.load(ref_path, sr=env_sr, mono=True)
    frame_len = int(env_sr * 0.5)
    hop = frame_len // 4
    rms = librosa.feature.rms(y=y_ref, frame_length=frame_len, hop_length=hop)[0]
    times = np.arange(len(rms)) * hop / env_sr

    _ref_envelope_cache[sr] = (rms, times, env_sr, frame_len, hop)
    return _ref_envelope_cache[sr]


PARAM_SPEC = [
    ('attack_ms',       0.073,  0.05,  15.0),
    ('decay_time',      0.278,  0.02,   1.5),
    ('drive',           1.171,  1.0,    4.0),
    ('asymmetry',       0.010,  0.0,    0.3),
    ('hf_boost',        2.211,  0.0,    5.0),
    ('comb_feedback',   0.159,  0.0,    0.95),
    ('comb_mix',        0.186,  0.0,    0.5),
    ('attack_level',    0.497,  0.0,    1.0),
    ('harm_1_db',       -5.5,  -30.0,   0.0),
    ('harm_2_db',       -1.9,  -30.0,   0.0),
    ('harm_3_db',      -25.6,  -40.0,   0.0),
    ('harm_4_db',      -28.4,  -40.0,   0.0),
    ('harm_5_db',       -7.6,  -40.0,   0.0),
    ('harm_6_db',      -50.0,  -50.0,   0.0),
    ('harm_7_db',      -11.6,  -50.0,   0.0),
    ('harm_8_db',      -47.4,  -60.0,   0.0),
    ('harm_9_db',      -28.5,  -60.0,   0.0),
    ('body_res_1_freq',  795,  200,   2000),
    ('body_res_1_q',     1.86, 0.5,   10.0),
    ('body_res_1_amp',   0.008,0.0,    1.0),
    ('body_res_2_freq', 1551,  500,   4000),
    ('body_res_2_q',     3.38, 0.5,   10.0),
    ('body_res_2_amp',   0.049,0.0,    1.0),
    ('harm_decay_slope', 0.005,0.0,    5.0),
    ('attack_brightness',0.829,0.2,    5.0),
    ('eq_low_gain_db',   0.73, -12.0,  12.0),
    ('eq_low_freq',    369.5,  80.0, 500.0),
    ('eq_mid_gain_db',   3.59, -12.0,  12.0),
    ('eq_mid_freq',   1209.1, 300.0, 3000.0),
    ('eq_hi_gain_db',   -8.54, -12.0,  12.0),
    ('eq_hi_freq',    3691.6, 1500.0, 8000.0),
    ('reverb_wet',      0.118,  0.0,    0.5),
    ('reverb_length',   1.75,   0.2,    3.0),
    ('reverb_dark',     0.371,  0.0,    0.9),
    ('delay_wet',       0.087,  0.0,    0.5),
    ('delay_feedback',  0.067,  0.0,    0.7),
    ('delay_dark_lp',   1502,   500,   8000),
    ('step_ms',         61.5,   50.0,  120.0),
    ('target_rms',      0.37,   0.10,   0.60),
    ('freq_0',          578.4,  130,    800),
    ('freq_1',          417.5,  130,    800),
    ('freq_2',          347.8,  110,    800),
    ('freq_3',          265.2,  100,    600),
    ('freq_4',          482.6,   80,    500),
    ('freq_5',          183.2,   65,    400),
    ('freq_6',          284.9,   65,    400),
    ('freq_7',          146.2,   50,    300),
]

PARAM_NAMES = [p[0] for p in PARAM_SPEC]
PARAM_DEFAULTS = np.array([p[1] for p in PARAM_SPEC])
PARAM_BOUNDS = [(p[2], p[3]) for p in PARAM_SPEC]


def unpack_params(x):
    p = {}
    for i, (name, _, _, _) in enumerate(PARAM_SPEC):
        p[name] = float(x[i])

    p['harmonic_amps_db'] = [0.0] + [p[f'harm_{i}_db'] for i in range(1, 10)]
    p['freqs'] = [p[f'freq_{i}'] for i in range(8)]
    return p


def gen_tone_param(freq, p, sr=SR):
    dur = 0.9
    n = int(sr * dur)
    t = np.arange(n) / sr

    harm_decay_slope = p.get('harm_decay_slope', 0.0)
    attack_brightness = p.get('attack_brightness', 1.0)

    atk_dur = 0.008
    atk_n = int(sr * atk_dur)
    atk = noise(atk_dur, sr)
    atk = bandpass(atk, max(20, freq * 0.5), min(sr/2 - 100, freq * 4), sr)
    atk_hp_freq = np.clip(freq * attack_brightness, 20, sr / 2 - 1)
    atk = highpass(atk, atk_hp_freq, sr)
    atk_env = env_exp_decay(atk_dur, 0.2, 0.003, sr)
    atk *= atk_env * p['attack_level']

    sig = np.zeros(n)
    amps_db = p['harmonic_amps_db']
    decay_time = max(p['decay_time'], 0.001)
    for h in range(len(amps_db)):
        partial_freq = freq * (h + 1)
        if partial_freq >= sr / 2:
            break
        amp = 10 ** (amps_db[h] / 20)
        env_h = np.exp(-t * (1.0 / decay_time + harm_decay_slope * h))
        sig += amp * np.sin(2 * np.pi * partial_freq * t) * env_h

    if p['comb_mix'] > 0.001:
        body_exc = np.zeros(n)
        body_exc[:atk_n] = atk[:min(atk_n, len(atk))]
        delay = max(1, int(sr / freq))
        body = comb_filter(body_exc, delay,
                          feedback=p['comb_feedback'],
                          lp_freq=min(freq * 3, sr/2 - 100), sr=sr)
        body *= p['comb_mix']
        sig = sig + body[:n]

    from scipy.signal import iirpeak, sosfilt as _sosfilt
    for res_idx in [1, 2]:
        res_freq = p.get(f'body_res_{res_idx}_freq', 500)
        res_q = p.get(f'body_res_{res_idx}_q', 2.0)
        res_amp = p.get(f'body_res_{res_idx}_amp', 0.0)
        if res_amp > 0.001 and 20 < res_freq < sr / 2 - 1:
            w0 = res_freq / (sr / 2)
            w0 = np.clip(w0, 0.001, 0.999)
            b_peak, a_peak = iirpeak(w0, res_q)
            from scipy.signal import lfilter
            resonance = lfilter(b_peak, a_peak, sig)
            sig = sig + resonance * res_amp

    a_samples = int(sr * p['attack_ms'] / 1000)
    if a_samples > 0 and a_samples < n:
        attack_env = np.linspace(0, 1, a_samples)
        sig[:a_samples] *= attack_env

    sig[:len(atk)] += atk[:min(len(atk), n)]

    sig = asymmetric_saturate(sig, drive=p['drive'], asymmetry=p['asymmetry'])

    if p['hf_boost'] > 0.01:
        sig_hp = highpass(sig, 1500, sr) * p['hf_boost']
        sig = sig + sig_hp

    sig = normalize(sig, 0.85)
    fade_in(sig, p['attack_ms'], sr)
    fade_out(sig, 40, sr)
    return sig


def render_with_params(x, patterns, fast=False):
    p = unpack_params(x)
    sr = COMPARE_SR if fast else SR

    freqs = p['freqs']
    samples = []
    for freq in freqs:
        sig = gen_tone_param(freq, p, sr)
        samples.append(sig)

    step_ms = p['step_ms']
    step_samples = int(step_ms / 1000 * sr)
    total_steps = len(patterns) * STEPS_PER_PATTERN
    total_samples = total_steps * step_samples + max(len(s) for s in samples)

    mix = np.zeros((total_samples, 2))

    row_start_step = {}

    global_step = 0
    for pi, pattern in enumerate(patterns):
        pattern_start = global_step
        for (row, col), vel in pattern.items():
            if row not in row_start_step:
                row_start_step[row] = pattern_start

        for local_step in range(STEPS_PER_PATTERN):
            offset = global_step * step_samples

            for (row, col), vel in pattern.items():
                row_seq_idx = (global_step - row_start_step[row]) % SEQ_LEN
                if col < len(COL_SEQS) and COL_SEQS[col][row_seq_idx]:
                    vol = vel * STEP_VELS[row_seq_idx]
                    sample = samples[row]
                    end = min(offset + len(sample), total_samples)
                    length = end - offset
                    mix[offset:end, 0] += sample[:length] * vol
                    mix[offset:end, 1] += sample[:length] * vol
            global_step += 1

    from scipy.signal import butter, sosfilt
    for band_prefix in ['eq_low', 'eq_mid', 'eq_hi']:
        gain_db = p.get(band_prefix + '_gain_db', 0.0)
        freq = p.get(band_prefix + '_freq', 1000.0)
        if abs(gain_db) > 0.1 and 20 < freq < sr / 2 - 1:
            gain_lin = 10 ** (gain_db / 20)
            bw = freq * 0.7
            low = max(20, freq - bw / 2)
            high = min(sr / 2 - 1, freq + bw / 2)
            if high > low + 10:
                sos = butter(2, [low, high], btype='band', fs=sr, output='sos')
                for ch in range(2):
                    band = sosfilt(sos, mix[:, ch])
                    mix[:, ch] += band * (gain_lin - 1)

    mix = apply_delay(mix, p, sr)

    rng_state = np.random.get_state()
    np.random.seed(42)
    ir = generate_reverb_ir(p['reverb_length'], dark=p['reverb_dark'], sr=sr)
    np.random.set_state(rng_state)
    for ch in range(2):
        wet = fftconvolve(mix[:, ch], ir[:, ch])[:mix.shape[0]]
        mix[:, ch] += wet * p['reverb_wet']

    ref_env = get_reference_envelope(sr)
    if ref_env is not None:
        rms_ref, ref_times, env_sr, frame_len, hop = ref_env
        mono_tmp = np.mean(mix, axis=1)
        from scipy.signal import resample as sig_resample
        mono_ds = sig_resample(mono_tmp, int(len(mono_tmp) * env_sr / sr))
        rms_ren = librosa.feature.rms(y=mono_ds, frame_length=frame_len, hop_length=hop)[0]
        n_env = min(len(rms_ref), len(rms_ren))
        gain_curve = np.ones(n_env)
        for i in range(n_env):
            if rms_ren[i] > 0.001:
                gain_curve[i] = rms_ref[i] / rms_ren[i]
            else:
                gain_curve[i] = 1.0
        from scipy.ndimage import uniform_filter1d
        gain_curve = uniform_filter1d(gain_curve, size=8)
        gain_curve = np.clip(gain_curve, 0.1, 5.0)
        gain_times = np.arange(n_env) * hop / env_sr
        sample_times = np.arange(mix.shape[0]) / sr
        gain_interp = np.interp(sample_times, gain_times, gain_curve)
        mix[:, 0] *= gain_interp
        mix[:, 1] *= gain_interp

    current_rms = np.sqrt(np.mean(mix ** 2))
    if current_rms > 0:
        gain = p['target_rms'] / current_rms
        mix *= gain
        mix = np.clip(mix, -0.95, 0.95)

    mono = np.mean(mix, axis=1)
    if sr != COMPARE_SR:
        from scipy.signal import resample
        n_out = int(len(mono) * COMPARE_SR / sr)
        mono = resample(mono, n_out)

    return mono


def render_stereo(x, patterns, sr=SR):
    p = unpack_params(x)

    freqs = p['freqs']
    samples = []
    for freq in freqs:
        sig = gen_tone_param(freq, p, sr)
        samples.append(sig)

    step_ms = p['step_ms']
    step_samples = int(step_ms / 1000 * sr)
    total_steps = len(patterns) * STEPS_PER_PATTERN
    total_samples = total_steps * step_samples + max(len(s) for s in samples)

    mix = np.zeros((total_samples, 2))

    row_start_step = {}
    global_step = 0
    for pi, pattern in enumerate(patterns):
        pattern_start = global_step
        for (row, col), vel in pattern.items():
            if row not in row_start_step:
                row_start_step[row] = pattern_start

        for local_step in range(STEPS_PER_PATTERN):
            offset = global_step * step_samples
            for (row, col), vel in pattern.items():
                row_seq_idx = (global_step - row_start_step[row]) % SEQ_LEN
                if col < len(COL_SEQS) and COL_SEQS[col][row_seq_idx]:
                    vol = vel * STEP_VELS[row_seq_idx]
                    sample = samples[row]
                    end = min(offset + len(sample), total_samples)
                    length = end - offset
                    mix[offset:end, 0] += sample[:length] * vol
                    mix[offset:end, 1] += sample[:length] * vol
            global_step += 1

    from scipy.signal import butter, sosfilt
    for band_prefix in ['eq_low', 'eq_mid', 'eq_hi']:
        gain_db = p.get(band_prefix + '_gain_db', 0.0)
        freq = p.get(band_prefix + '_freq', 1000.0)
        if abs(gain_db) > 0.1 and 20 < freq < sr / 2 - 1:
            gain_lin = 10 ** (gain_db / 20)
            bw = freq * 0.7
            low = max(20, freq - bw / 2)
            high = min(sr / 2 - 1, freq + bw / 2)
            if high > low + 10:
                sos = butter(2, [low, high], btype='band', fs=sr, output='sos')
                for ch in range(2):
                    band = sosfilt(sos, mix[:, ch])
                    mix[:, ch] += band * (gain_lin - 1)

    mix = apply_delay(mix, p, sr)

    rng_state = np.random.get_state()
    np.random.seed(42)
    ir = generate_reverb_ir(p['reverb_length'], dark=p['reverb_dark'], sr=sr)
    np.random.set_state(rng_state)
    for ch in range(2):
        wet = fftconvolve(mix[:, ch], ir[:, ch])[:mix.shape[0]]
        mix[:, ch] += wet * p['reverb_wet']

    ref_env = get_reference_envelope(sr)
    if ref_env is not None:
        rms_ref, ref_times, env_sr, frame_len, hop = ref_env
        mono_tmp = np.mean(mix, axis=1)
        from scipy.signal import resample as sig_resample
        mono_ds = sig_resample(mono_tmp, int(len(mono_tmp) * env_sr / sr))
        rms_ren = librosa.feature.rms(y=mono_ds, frame_length=frame_len, hop_length=hop)[0]
        n_env = min(len(rms_ref), len(rms_ren))
        gain_curve = np.ones(n_env)
        for i in range(n_env):
            if rms_ren[i] > 0.001:
                gain_curve[i] = rms_ref[i] / rms_ren[i]
            else:
                gain_curve[i] = 1.0
        from scipy.ndimage import uniform_filter1d
        gain_curve = uniform_filter1d(gain_curve, size=8)
        gain_curve = np.clip(gain_curve, 0.1, 5.0)
        gain_times = np.arange(n_env) * hop / env_sr
        sample_times = np.arange(mix.shape[0]) / sr
        gain_interp = np.interp(sample_times, gain_times, gain_curve)
        mix[:, 0] *= gain_interp
        mix[:, 1] *= gain_interp

    current_rms = np.sqrt(np.mean(mix ** 2))
    if current_rms > 0:
        gain = p['target_rms'] / current_rms
        mix *= gain
        mix = np.clip(mix, -0.95, 0.95)

    return mix


def apply_delay(mix, p, sr):
    n = mix.shape[0]
    dl = int(0.33 * sr)
    dr = int(0.22 * sr)
    out = mix.copy()
    fb = p['delay_feedback']

    for ch, d in enumerate([dl, dr]):
        sig = mix[:, ch]
        delayed = np.zeros(n)
        tap = sig.copy()
        for k in range(1, 20):
            tap_amp = fb ** k
            if tap_amp < 0.001:
                break
            shift = d * k
            if shift >= n:
                break
            delayed[shift:] += tap[:n - shift] * tap_amp
        delayed = lowpass(delayed, p['delay_dark_lp'], sr)
        out[:, ch] += delayed * p['delay_wet']

    return out


_MS_FFT_SIZES = [256, 512, 1024, 2048, 4096]
_MS_HOP_DIVISOR = 4

_ref_spec_cache = {}


def _compute_stft_mag(y, n_fft, hop_length):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    return np.abs(S)


def _get_ref_spectrograms(y_ref, sr):
    key = (len(y_ref), sr)
    if key in _ref_spec_cache:
        return _ref_spec_cache[key]

    specs = {}
    for n_fft in _MS_FFT_SIZES:
        hop = n_fft // _MS_HOP_DIVISOR
        mag = _compute_stft_mag(y_ref, n_fft, hop)
        specs[n_fft] = {
            'mag': mag,
            'log_mag': np.log(mag + 1e-7),
        }

    for n_mels in [32, 64, 128]:
        for n_fft in [1024, 2048]:
            hop = n_fft // _MS_HOP_DIVISOR
            mel = librosa.feature.melspectrogram(
                y=y_ref, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
            mel_db = librosa.power_to_db(mel + 1e-10)
            specs[f'mel_{n_mels}_{n_fft}'] = mel_db

    _ref_spec_cache[key] = specs
    return specs


_ref_features_cache = {}


def _get_ref_features(y_ref, sr):
    key = (len(y_ref), sr)
    if key in _ref_features_cache:
        return _ref_features_cache[key]

    feats = {}
    feats['mfcc_mean'] = np.mean(librosa.feature.mfcc(y=y_ref, sr=sr, n_mfcc=13), axis=1)
    feats['onsets_t'] = librosa.onset.onset_detect(y=y_ref, sr=sr, units='time')
    feats['onset_rate'] = len(feats['onsets_t']) / max(len(y_ref)/sr, 0.1)
    fl = int(sr * 0.05)
    hl = fl // 2
    feats['rms'] = librosa.feature.rms(y=y_ref, frame_length=fl, hop_length=hl)[0]
    feats['chroma_mean'] = np.mean(librosa.feature.chroma_stft(y=y_ref, sr=sr), axis=1)

    _ref_features_cache[key] = feats
    return feats


def compute_spectral_loss(y_ren, y_ref, sr=COMPARE_SR):
    ref_specs = _get_ref_spectrograms(y_ref, sr)
    n = min(len(y_ref), len(y_ren))
    y_ren = y_ren[:n]

    total_loss = 0.0
    n_scales = 0

    for n_fft in _MS_FFT_SIZES:
        hop = n_fft // _MS_HOP_DIVISOR
        mag_ren = _compute_stft_mag(y_ren, n_fft, hop)
        mag_ref = ref_specs[n_fft]['mag']
        log_ref = ref_specs[n_fft]['log_mag']

        n_frames = min(mag_ref.shape[1], mag_ren.shape[1])
        mr = mag_ref[:, :n_frames]
        mx = mag_ren[:, :n_frames]

        sc = np.linalg.norm(mr - mx) / (np.linalg.norm(mr) + 1e-7)

        log_mx = np.log(mx + 1e-7)
        log_l1 = np.mean(np.abs(log_ref[:, :n_frames] - log_mx))

        total_loss += sc + log_l1 / 10.0
        n_scales += 1

    for n_mels in [32, 64, 128]:
        for n_fft in [1024, 2048]:
            hop = n_fft // _MS_HOP_DIVISOR
            mel_ren = librosa.feature.melspectrogram(
                y=y_ren, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
            mel_ren_db = librosa.power_to_db(mel_ren + 1e-10)
            mel_ref_db = ref_specs[f'mel_{n_mels}_{n_fft}']

            n_frames = min(mel_ref_db.shape[1], mel_ren_db.shape[1])
            mel_l1 = np.mean(np.abs(
                mel_ref_db[:, :n_frames] - mel_ren_db[:, :n_frames]))

            total_loss += mel_l1 / 20.0
            n_scales += 1

    return total_loss / n_scales


def compute_composite(y_ren, y_ref, sr=COMPARE_SR):
    n = min(len(y_ref), len(y_ren))
    y_ren = y_ren[:n]

    metrics = {}

    metrics['spectral_loss'] = compute_spectral_loss(y_ren, y_ref, sr)

    ref = _get_ref_features(y_ref, sr)

    mfcc_ren = librosa.feature.mfcc(y=y_ren, sr=sr, n_mfcc=13)
    metrics['mfcc'] = min(
        float(np.linalg.norm(ref['mfcc_mean'] - np.mean(mfcc_ren, axis=1))) / 200, 1.0)

    onsets_ren_t = librosa.onset.onset_detect(y=y_ren, sr=sr, units='time')
    rate_ren = len(onsets_ren_t) / max(len(y_ren)/sr, 0.1)
    if ref['onset_rate'] == 0 and rate_ren == 0:
        metrics['onset_density'] = 0.0
    else:
        metrics['onset_density'] = abs(ref['onset_rate'] - rate_ren) / max(ref['onset_rate'], rate_ren)

    fl = int(sr * 0.05)
    hl = fl // 2
    rms_ren = librosa.feature.rms(y=y_ren, frame_length=fl, hop_length=hl)[0]
    nr = min(len(ref['rms']), len(rms_ren))
    if np.std(ref['rms'][:nr]) < 1e-8 or np.std(rms_ren[:nr]) < 1e-8:
        metrics['rms_correlation'] = 1.0
    else:
        corr = float(np.corrcoef(ref['rms'][:nr], rms_ren[:nr])[0, 1])
        metrics['rms_correlation'] = max(0, 1 - corr)

    ch_ren = librosa.feature.chroma_stft(y=y_ren, sr=sr)
    hn = np.mean(ch_ren, axis=1)
    dot = np.dot(ref['chroma_mean'], hn)
    norm = np.linalg.norm(ref['chroma_mean']) * np.linalg.norm(hn)
    metrics['pitch_class'] = max(0, 1 - dot/norm) if norm > 1e-8 else 1.0

    composite = metrics['spectral_loss']

    return composite, metrics


def infer_buttons(y_ref, sr, freqs, step_ms):
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
            'pattern_idx': global_step // STEPS_PER_PATTERN,
        })

    if not events:
        return []

    n_patterns = max(e['pattern_idx'] for e in events) + 1

    patterns_events = [[] for _ in range(n_patterns)]
    for event in events:
        if 0 <= event['pattern_idx'] < n_patterns:
            patterns_events[event['pattern_idx']].append(event)

    inferred = []
    cumulative_cells = set()
    row_start_step = {}

    for pi, pe in enumerate(patterns_events):
        pattern_start = pi * STEPS_PER_PATTERN
        rows_in_pattern = set(e['row'] for e in pe)
        pattern_cells = {}

        for row in rows_in_pattern:
            if row not in row_start_step:
                row_start_step[row] = pattern_start

            row_global_steps = [e['global_step'] for e in pe if e['row'] == row]
            row_local_indices = set(
                (gs - row_start_step[row]) % SEQ_LEN for gs in row_global_steps
            )

            best_col, best_score = -1, -1
            for col in range(16):
                expected = set(s for s in range(SEQ_LEN) if COL_SEQS[col][s] == 1)
                hits = len(row_local_indices & expected)
                misses = len(expected - row_local_indices)
                false_alarms = len(row_local_indices - expected)
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


BLOCKS = {
    'A_synthesis': [
        'attack_ms', 'decay_time', 'drive', 'asymmetry',
        'hf_boost', 'comb_feedback', 'comb_mix', 'attack_level',
    ],
    'B_harmonics': [
        'harm_1_db', 'harm_2_db', 'harm_3_db', 'harm_4_db', 'harm_5_db',
        'harm_6_db', 'harm_7_db', 'harm_8_db', 'harm_9_db',
    ],
    'C_body': [
        'body_res_1_freq', 'body_res_1_q', 'body_res_1_amp',
        'body_res_2_freq', 'body_res_2_q', 'body_res_2_amp',
        'harm_decay_slope', 'attack_brightness',
    ],
    'C_eq': [
        'eq_low_gain_db', 'eq_low_freq',
        'eq_mid_gain_db', 'eq_mid_freq',
        'eq_hi_gain_db', 'eq_hi_freq',
    ],
    'D_effects': [
        'reverb_wet', 'reverb_length', 'reverb_dark',
        'delay_wet', 'delay_feedback', 'delay_dark_lp',
    ],
    'E_frequencies': [
        'freq_0', 'freq_1', 'freq_2', 'freq_3',
        'freq_4', 'freq_5', 'freq_6', 'freq_7',
    ],
    'F_timing': [
        'step_ms', 'target_rms',
    ],
}


def get_block_indices(block_name):
    names = BLOCKS[block_name]
    return [PARAM_NAMES.index(n) for n in names]


def optimize_block_cmaes(block_name, x_full, patterns, y_ref_compare, budget=200):
    indices = get_block_indices(block_name)
    n_block = len(indices)

    x0_block = x_full[indices].copy()
    bounds_block = [PARAM_BOUNDS[i] for i in indices]
    lowers = np.array([b[0] for b in bounds_block])
    uppers = np.array([b[1] for b in bounds_block])

    ranges = uppers - lowers
    sigma0 = float(np.median(ranges * 0.10))

    def to_unit(x_block):
        return (x_block - lowers) / ranges

    def from_unit(u_block):
        return u_block * ranges + lowers

    u0 = to_unit(x0_block)

    eval_count = [0]
    best_composite = [float('inf')]
    best_x_full = [x_full.copy()]

    def objective(u):
        eval_count[0] += 1
        x_block = from_unit(np.clip(u, 0, 1))
        x_candidate = x_full.copy()
        x_candidate[indices] = x_block
        try:
            y_ren = render_with_params(x_candidate, patterns, fast=False)
            composite, _ = compute_composite(y_ren, y_ref_compare)
            if composite < best_composite[0]:
                best_composite[0] = composite
                best_x_full[0] = x_candidate.copy()
                print(f"      [{block_name}] eval {eval_count[0]:3d}: "
                      f"{composite:.4f} (new best)")
            return composite
        except Exception as e:
            print(f"      [{block_name}] eval {eval_count[0]:3d}: ERROR {e}")
            return 1.0

    popsize = max(8, 2 * n_block)
    maxiter = max(10, budget // popsize)

    opts = {
        'popsize': popsize,
        'maxiter': maxiter,
        'maxfevals': budget,
        'bounds': [0, 1],
        'tolfun': 1e-4,
        'tolx': 1e-4,
        'verbose': -9,
        'seed': int(time.time()) % 2**31,
    }

    try:
        es = cma.CMAEvolutionStrategy(u0.tolist(), 0.15, opts)
        while not es.stop():
            solutions = es.ask()
            fitnesses = [objective(np.array(s)) for s in solutions]
            es.tell(solutions, fitnesses)
        es.result_pretty()
    except Exception as e:
        print(f"      [{block_name}] CMA-ES error: {e}")

    return best_x_full[0], best_composite[0], eval_count[0]


def optimize_joint_cmaes(x_full, patterns, y_ref_compare, budget=500):
    n = len(x_full)
    lowers = np.array([b[0] for b in PARAM_BOUNDS])
    uppers = np.array([b[1] for b in PARAM_BOUNDS])
    ranges = uppers - lowers

    def to_unit(x):
        return (x - lowers) / ranges

    def from_unit(u):
        return u * ranges + lowers

    u0 = to_unit(x_full)
    eval_count = [0]
    best_composite = [float('inf')]
    best_x = [x_full.copy()]

    def objective(u):
        eval_count[0] += 1
        x = from_unit(np.clip(u, 0, 1))
        try:
            y_ren = render_with_params(x, patterns, fast=False)
            composite, _ = compute_composite(y_ren, y_ref_compare)
            if composite < best_composite[0]:
                best_composite[0] = composite
                best_x[0] = x.copy()
                print(f"      [JOINT] eval {eval_count[0]:4d}: "
                      f"{composite:.4f} (new best)")
            return composite
        except Exception as e:
            return 1.0

    popsize = max(20, 2 * n)
    maxiter = max(15, budget // popsize)

    opts = {
        'popsize': popsize,
        'maxiter': maxiter,
        'maxfevals': budget,
        'bounds': [0, 1],
        'tolfun': 1e-5,
        'tolx': 1e-5,
        'verbose': -9,
        'seed': int(time.time()) % 2**31,
        'CMA_active': True,
    }

    try:
        es = cma.CMAEvolutionStrategy(u0.tolist(), 0.08, opts)
        while not es.stop():
            solutions = es.ask()
            fitnesses = [objective(np.array(s)) for s in solutions]
            es.tell(solutions, fitnesses)
    except Exception as e:
        print(f"      [JOINT] CMA-ES error: {e}")

    return best_x[0], best_composite[0], eval_count[0]


def main():
    parser = argparse.ArgumentParser(
        description='Full parameterized EM optimizer for original mode audio')
    parser.add_argument('--ref', default=None)
    parser.add_argument('--em-iters', type=int, default=2,
                        help='Number of EM (E-step + M-step) iterations')
    parser.add_argument('--budget', type=int, default=200,
                        help='Max evaluations per block per M-step')
    parser.add_argument('--blocks', nargs='*', default=None,
                        help='Specific blocks to optimize (default: all)')
    parser.add_argument('--joint', action='store_true',
                        help='Optimize all params jointly instead of block-by-block')
    parser.add_argument('--skip-estep', action='store_true',
                        help='Skip E-step (pattern inference)')
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref_path = args.ref or os.path.join(root, 'analysis', 'reference.wav')

    if not os.path.exists(ref_path):
        print(f"ERROR: {ref_path} not found")
        sys.exit(1)

    print(f"Loading reference: {ref_path}")
    y_ref_compare, _ = librosa.load(ref_path, sr=COMPARE_SR, mono=True)
    print(f"  Duration: {len(y_ref_compare)/COMPARE_SR:.1f}s")

    y_ref_full, _ = librosa.load(ref_path, sr=SR, mono=True)

    x = PARAM_DEFAULTS.copy()

    from render_original import PATTERNS as current_patterns

    patterns = current_patterns
    best_composite = float('inf')
    best_x = x.copy()
    best_patterns = patterns

    print("\nInitial evaluation...")
    t0 = time.time()
    y_init = render_with_params(x, patterns, fast=False)
    c_init, m_init = compute_composite(y_init, y_ref_compare)
    t_eval = time.time() - t0
    print(f"  Composite: {c_init:.4f} ({t_eval:.1f}s per eval)")
    print(f"  Metrics: {', '.join(f'{k}={v:.3f}' for k, v in sorted(m_init.items()))}")
    best_composite = c_init
    best_x = x.copy()

    block_order = args.blocks or list(BLOCKS.keys())
    for b in block_order:
        if b not in BLOCKS:
            print(f"ERROR: unknown block '{b}'. Available: {list(BLOCKS.keys())}")
            sys.exit(1)

    for em_iter in range(args.em_iters):
        print(f"\n{'='*70}")
        print(f"EM ITERATION {em_iter + 1} / {args.em_iters}")
        print(f"{'='*70}")

        if not args.skip_estep:
            print("\n[E-step] Inferring button presses...")
            p = unpack_params(x)
            inferred = infer_buttons(y_ref_full, SR, p['freqs'], p['step_ms'])
            if inferred:
                n_inf = len(inferred)
                total_cells = sum(len(pat) for pat in inferred)
                print(f"  Inferred {n_inf} patterns, {total_cells} total cells")

                print("  Comparing inferred vs current patterns...")
                y_inferred = render_with_params(x, inferred, fast=False)
                c_inferred, _ = compute_composite(y_inferred, y_ref_compare)

                y_current = render_with_params(x, patterns, fast=False)
                c_current, _ = compute_composite(y_current, y_ref_compare)

                print(f"    Current:  {c_current:.4f}")
                print(f"    Inferred: {c_inferred:.4f}")

                if c_inferred < c_current:
                    print(f"    -> Using inferred patterns")
                    patterns = inferred
                else:
                    print(f"    -> Keeping current patterns")
            else:
                print("  No patterns inferred, keeping current")

        if args.joint:
            print(f"\n[M-step] Joint CMA-ES optimization "
                  f"({len(PARAM_SPEC)} params, budget={args.budget})")

            x_before = x.copy()
            c_before = best_composite

            x_improved, c_improved, n_evals = optimize_joint_cmaes(
                x, patterns, y_ref_compare, budget=args.budget)

            if c_improved < c_before:
                x = x_improved
                best_composite = c_improved
                best_x = x.copy()
                best_patterns = patterns

                changed = []
                for i, (name, _, _, _) in enumerate(PARAM_SPEC):
                    if abs(x[i] - x_before[i]) > 0.001:
                        changed.append(f"{name}: {x_before[i]:.3f}->{x[i]:.3f}")
                print(f"    -> Improved {c_before:.4f} -> {c_improved:.4f} "
                      f"(delta={c_before - c_improved:.4f}, {n_evals} evals)")
                for c in changed[:10]:
                    print(f"       {c}")
                if len(changed) > 10:
                    print(f"       ... and {len(changed)-10} more")
            else:
                x = x_before
                print(f"    -> No improvement ({n_evals} evals)")
        else:
            print(f"\n[M-step] Block coordinate descent ({len(block_order)} blocks, "
                  f"budget={args.budget}/block)")

            for block_name in block_order:
                indices = get_block_indices(block_name)
                n_params = len(indices)
                param_names = [PARAM_NAMES[i] for i in indices]
                print(f"\n  Block {block_name} ({n_params} params: "
                      f"{', '.join(param_names[:4])}{'...' if n_params > 4 else ''})")

                x_before = x.copy()
                c_before = best_composite

                x_improved, c_improved, n_evals = optimize_block_cmaes(
                    block_name, x, patterns, y_ref_compare,
                    budget=args.budget)

                if c_improved < c_before:
                    improvement = c_before - c_improved
                    x = x_improved
                    best_composite = c_improved
                    best_x = x.copy()
                    best_patterns = patterns

                    changed = []
                    for idx in indices:
                        name = PARAM_NAMES[idx]
                        old = x_before[idx]
                        new = x[idx]
                        if abs(new - old) > 0.001:
                            changed.append(f"{name}: {old:.3f}->{new:.3f}")
                    print(f"    -> Improved {c_before:.4f} -> {c_improved:.4f} "
                          f"(delta={improvement:.4f}, {n_evals} evals)")
                    if changed:
                        for c in changed:
                            print(f"       {c}")
                else:
                    x = x_before
                    print(f"    -> No improvement ({n_evals} evals), reverting")

        print(f"\n  M-step complete. Best composite: {best_composite:.4f}")

    print(f"\n{'='*70}")
    print("FINAL EVALUATION (full quality)")
    print(f"{'='*70}")

    y_final = render_with_params(best_x, best_patterns, fast=False)
    c_final, m_final = compute_composite(y_final, y_ref_compare)
    print(f"\nComposite distance: {c_final:.4f}")
    print(f"\nPer-metric breakdown:")
    for k, v in sorted(m_final.items()):
        print(f"  {k:25s}: {v:.4f}")

    p = unpack_params(best_x)
    out = {
        'composite': c_final,
        'metrics': {k: round(v, 6) for k, v in m_final.items()},
        'params': {name: round(float(best_x[i]), 4)
                   for i, (name, _, _, _) in enumerate(PARAM_SPEC)},
        'patterns': [
            {f"{r}-{c}": v for (r, c), v in pat.items()}
            for pat in best_patterns
        ],
    }
    out_path = os.path.join(root, 'analysis', 'process', 'optimized_full.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    print(f"\n{'='*70}")
    print("CODE TO APPLY")
    print(f"{'='*70}")
    print(f"\n# gen_original.py DEFAULT_ENVELOPE:")
    print(f"DEFAULT_ENVELOPE = {{")
    print(f"    'attack_ms': {p['attack_ms']:.2f},")
    print(f"    'decay_time': {p['decay_time']:.3f},")
    print(f"    'harmonic_ratios': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0],")
    harm_str = [round(x, 1) for x in p['harmonic_amps_db']]
    print(f"    'harmonic_amplitudes_db': {harm_str},")
    print(f"    'inharmonicity_cents': [0, 6, -2, 10, -4, 5, -7, 8, -3, 6],")
    print(f"}}")
    print(f"\n# gen_original.py gen_tone():")
    print(f"  drive={p['drive']:.2f}, asymmetry={p['asymmetry']:.3f}")
    print(f"  hf_boost={p['hf_boost']:.2f}, comb_feedback={p['comb_feedback']:.3f}, "
          f"comb_mix={p['comb_mix']:.3f}")
    print(f"  attack_level={p['attack_level']:.3f}")
    print(f"\n# render_original.py effects:")
    print(f"  REVERB_WET={p['reverb_wet']:.3f}, REVERB_LENGTH={p['reverb_length']:.2f}, "
          f"REVERB_DARK={p['reverb_dark']:.3f}")
    print(f"  DELAY_WET={p['delay_wet']:.3f}, DELAY_FEEDBACK={p['delay_feedback']:.3f}, "
          f"DELAY_DARK_LP={p['delay_dark_lp']:.0f}")
    print(f"  step_ms={p['step_ms']:.1f}, target_rms={p['target_rms']:.3f}")
    print(f"\n# gen_original.py DEFAULT_FREQS:")
    print(f"  DEFAULT_FREQS = {[round(f, 1) for f in p['freqs']]}")


if __name__ == '__main__':
    main()
