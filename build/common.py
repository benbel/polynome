"""Shared DSP library for audio generation."""

import numpy as np
from scipy.signal import sosfilt, butter
from scipy.io import wavfile
import subprocess, json, os

SR = 44100


# ======================== OSCILLATORS ========================

def sine(freq, dur, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t)


def saw(freq, dur, sr=SR, num_harmonics=30):
    t = np.arange(int(sr * dur)) / sr
    out = np.zeros_like(t)
    for k in range(1, num_harmonics + 1):
        if k * freq > sr / 2:
            break
        out += ((-1) ** (k + 1)) * np.sin(2 * np.pi * k * freq * t) / k
    return out * (2 / np.pi)


def pulse(freq, dur, pw=0.5, sr=SR, num_harmonics=30):
    t = np.arange(int(sr * dur)) / sr
    out = np.zeros_like(t)
    for k in range(1, num_harmonics + 1):
        if k * freq > sr / 2:
            break
        out += np.sin(np.pi * k * pw) * np.sin(2 * np.pi * k * freq * t) / k
    return out * (4 / np.pi)


def noise(dur, sr=SR):
    return np.random.uniform(-1, 1, int(sr * dur))


def stereo_noise(dur, sr=SR):
    return np.column_stack([noise(dur, sr), noise(dur, sr)])


# ======================== ENVELOPES ========================

def env_adsr(dur, attack, decay, sustain, release, sr=SR):
    n = int(sr * dur)
    a = int(sr * attack)
    d = int(sr * decay)
    r = int(sr * release)
    s = max(0, n - a - d - r)
    env = np.concatenate([
        np.linspace(0, 1, max(a, 1)),
        np.linspace(1, sustain, max(d, 1)),
        np.full(s, sustain),
        np.linspace(sustain, 0, max(r, 1)),
    ])
    return env[:n]


def env_exp_decay(dur, attack_ms, decay_time, sr=SR):
    n = int(sr * dur)
    a = int(sr * attack_ms / 1000)
    env = np.zeros(n)
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    t = np.arange(n - a) / sr
    env[a:] = np.exp(-t / max(decay_time, 0.001))
    return env


# ======================== FILTERS ========================

def lowpass(sig, freq, sr=SR, order=4):
    freq = np.clip(freq, 20, sr / 2 - 1)
    sos = butter(order, freq, btype='low', fs=sr, output='sos')
    return sosfilt(sos, sig)


def highpass(sig, freq, sr=SR, order=4):
    freq = np.clip(freq, 20, sr / 2 - 1)
    sos = butter(order, freq, btype='high', fs=sr, output='sos')
    return sosfilt(sos, sig)


def bandpass(sig, low, high, sr=SR, order=2):
    low = np.clip(low, 20, sr / 2 - 1)
    high = np.clip(high, low + 1, sr / 2 - 1)
    sos = butter(order, [low, high], btype='band', fs=sr, output='sos')
    return sosfilt(sos, sig)


def filter_sweep(sig, start_freq, end_freq, sr=SR, order=4, block_size=256):
    n = len(sig)
    out = np.zeros(n)
    freqs = np.logspace(np.log10(max(start_freq, 20)), np.log10(min(end_freq, sr / 2 - 100)),
                        n // block_size + 1)
    freqs = np.clip(freqs, 20, sr / 2 - 100)
    zi = None
    for i, fc in enumerate(freqs):
        start = i * block_size
        end = min(start + block_size, n)
        if start >= n:
            break
        sos = butter(order, fc, btype='low', fs=sr, output='sos')
        if zi is None:
            block, zi = sosfilt(sos, sig[start:end], zi=np.zeros((sos.shape[0], 2)))
        else:
            block, zi = sosfilt(sos, sig[start:end], zi=zi)
        out[start:end] = block
    return out


def moog_ladder(sig, cutoff_hz, resonance, sr=SR):
    """4-pole resonant ladder filter (Huovilainen model)."""
    if isinstance(cutoff_hz, (int, float)):
        cutoff_hz = np.full(len(sig), cutoff_hz)
    s = [0.0, 0.0, 0.0, 0.0]
    out = np.zeros(len(sig))
    for i in range(len(sig)):
        g = 1 - np.exp(-2 * np.pi * cutoff_hz[i] / sr)
        feedback = resonance * np.tanh(s[3])
        x = np.tanh(sig[i] - feedback)
        for stage in range(4):
            s[stage] += g * (x - s[stage])
            x = s[stage]
        out[i] = s[3]
    return out


# ======================== WAVESHAPING ========================

def wavefold(sig, folds=3):
    x = sig * folds
    return (4 / np.pi) * np.arcsin(np.sin(np.pi * x / 2))


def tanh_saturate(sig, drive=4):
    return np.tanh(sig * drive) / np.tanh(drive)


def asymmetric_saturate(sig, drive=2, asymmetry=0.15):
    pos = np.tanh(sig * drive * (1 + asymmetry)) / (1 + asymmetry)
    neg = np.tanh(sig * drive * (1 - asymmetry)) / (1 - asymmetry)
    return np.where(sig >= 0, pos, neg)


def bitcrush(sig, bits=10, downsample=3):
    step = 2 ** (-(bits - 1))
    crushed = np.round(sig / step) * step
    if downsample > 1:
        held = crushed.copy()
        for i in range(len(held)):
            if i % downsample != 0:
                held[i] = held[i - 1]
        return held
    return crushed


def comb_filter(sig, delay_samples, feedback=0.7, lp_freq=None, sr=SR):
    n = len(sig)
    out = np.zeros(n)
    buf = np.zeros(delay_samples)
    p = 0
    for i in range(n):
        out[i] = sig[i] + buf[p] * feedback
        buf[p] = out[i]
        if lp_freq and p > 0:
            alpha = 1 - np.exp(-2 * np.pi * lp_freq / sr)
            buf[p] = alpha * buf[p] + (1 - alpha) * buf[(p - 1) % delay_samples]
        p = (p + 1) % delay_samples
    return out


# ======================== UTILITIES ========================

def normalize(sig, peak=0.85):
    mx = np.max(np.abs(sig))
    if mx > 0:
        sig = sig * (peak / mx)
    return sig


def fade_in(sig, ms, sr=SR):
    n = int(sr * ms / 1000)
    if n > 0:
        sig[:n] *= np.linspace(0, 1, n)
    return sig


def fade_out(sig, ms, sr=SR):
    n = int(sr * ms / 1000)
    if n > 0:
        sig[-n:] *= np.linspace(1, 0, n)
    return sig


def to_stereo(mono, pan=0):
    l = mono * np.sqrt((1 - pan) / 2)
    r = mono * np.sqrt((1 + pan) / 2)
    return np.column_stack([l, r])


def mix_stereo(*signals):
    maxlen = max(len(s) for s in signals)
    out = np.zeros((maxlen, 2))
    for s in signals:
        if s.ndim == 1:
            s = to_stereo(s, 0)
        out[:len(s)] += s
    return out


def export_ogg(stereo_signal, path, sr=SR):
    wav_path = path.replace('.ogg', '.wav')
    sig16 = np.clip(stereo_signal, -1, 1)
    sig16 = (sig16 * 32767).astype(np.int16)
    wavfile.write(wav_path, sr, sig16)
    if path.endswith('.ogg'):
        subprocess.run([
            'ffmpeg', '-y', '-i', wav_path,
            '-c:a', 'libvorbis', '-q:a', '3', path
        ], capture_output=True)
        os.remove(wav_path)


def write_manifest(out_dir, instruments, fx_config, fmt='ogg'):
    manifest = {
        'format': fmt,
        'instruments': [
            {
                'id': inst['id'], 'label': inst['label'],
                'pitchCount': inst['pitchCount'],
                'type': inst.get('type', 'main'),
            }
            for inst in instruments
        ],
        'fx': fx_config,
    }
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)


def generate_reverb_ir(length_s, dark=0.8, sr=SR):
    """Generate a reverb impulse response."""
    n = int(sr * length_s)
    ir = np.zeros((n, 2))
    for ch in range(2):
        d = np.random.randn(n) * np.power(np.linspace(1, 0, n), 2.5)
        for i in range(1, n):
            d[i] = d[i] * (1 - dark) + d[i - 1] * dark
        ir[:, ch] = d
    return normalize(ir, 0.85)
