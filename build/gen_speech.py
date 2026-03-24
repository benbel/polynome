"""Generate speech mode audio assets.

Phoneme-driven percussive speech. Uses pure formant synthesis
(no external TTS dependency) to generate phoneme-like sounds.
Each phoneme class becomes an instrument with pitch variants.
"""

import os
import numpy as np
from common import (
    SR, sine, noise,
    env_adsr, env_exp_decay, lowpass, highpass, bandpass,
    normalize, fade_in, fade_out,
    to_stereo, export_ogg, write_manifest, generate_reverb_ir,
)

FREQS = [440, 392, 330, 294, 262, 220, 196, 165]
MEL_FREQS = [523, 440, 392, 330, 262, 220, 196, 165]

DURATIONS = {
    'vowels': 0.4, 'clicks': 0.15, 'hiss': 0.3,
    'nasal': 0.35, 'liquid': 0.3, 'syllables': 0.4,
}


def glottal_source(f0, dur, sr=SR):
    """Simple glottal source for speech synthesis."""
    n = int(sr * dur)
    phase = np.cumsum(np.full(n, f0) / sr)
    cycle_pos = phase % 1
    oq = 0.6
    pulse = np.where(cycle_pos < oq, 3 * (cycle_pos / oq) ** 2 - 2 * (cycle_pos / oq) ** 3, 0)
    return np.diff(pulse, prepend=0)


def formant_synth(source, formants, sr=SR):
    """Apply formant filter bank. formants: list of (freq, bw, gain_db)."""
    out = np.zeros_like(source)
    for f, bw, gain in formants:
        low = max(20, f - bw / 2)
        high = min(sr / 2 - 100, f + bw / 2)
        if low >= high:
            continue
        filtered = bandpass(source, low, high, sr, order=2)
        out += filtered * (10 ** (gain / 20))
    return out


# ======================== INSTRUMENTS ========================

def gen_vowels(freq, sr=SR):
    """Tonal vowel sounds — pitched."""
    dur = DURATIONS['vowels']
    n = int(sr * dur)

    # Cycle through vowel formants based on pitch (different vowel per pitch)
    vowels = [
        [(730, 90, 0), (1090, 110, -6), (2440, 170, -12)],   # a
        [(270, 60, 0), (2290, 200, -6), (3010, 200, -12)],    # i
        [(300, 60, 0), (870, 100, -6), (2240, 170, -12)],     # u
        [(530, 80, 0), (1840, 150, -6), (2480, 180, -12)],    # e
        [(570, 80, 0), (840, 100, -6), (2410, 170, -12)],     # o
    ]
    # Pick vowel based on frequency hash
    vidx = int(freq) % len(vowels)
    source = glottal_source(freq, dur, sr)
    sig = formant_synth(source, vowels[vidx], sr)

    # Layer 3 copies for thickness
    for offset_ms in [5, -5]:
        offset = int(sr * offset_ms / 1000)
        shifted = np.zeros(n)
        src = sig[:n]
        if offset > 0:
            shifted[offset:] = src[:n - offset]
        else:
            shifted[:n + offset] = src[-offset:]
        sig[:n] = sig[:n] + shifted * 0.3

    sig = sig[:n]
    env = env_adsr(dur, 0.01, 0.05, 0.7, 0.1, sr)
    sig *= env[:len(sig)]

    result = normalize(sig, 0.85)
    fade_in(result, 5, sr)
    fade_out(result, 30, sr)
    return to_stereo(result, 0)


def gen_clicks(freq, sr=SR):
    """Percussive plosive sounds."""
    dur = DURATIONS['clicks']
    n = int(sr * dur)

    # Short noise burst shaped by formant
    burst = noise(0.01, sr) * 0.8
    burst_pad = np.zeros(n)
    burst_pad[:len(burst)] = burst

    # Formant shift based on frequency
    f1 = 200 + (freq / 440) * 300
    sig = bandpass(burst_pad, max(20, f1 - 100), min(sr/2 - 100, f1 + 200), sr)

    # Short tonal tail
    tail = sine(freq * 0.5, 0.05, sr) * 0.3
    tail_env = env_exp_decay(0.05, 0.5, 0.02, sr)
    tail *= tail_env
    sig[:len(tail)] += tail

    env = env_exp_decay(dur, 0.5, 0.05, sr)
    sig *= env[:len(sig)]

    result = normalize(sig, 0.85)
    return to_stereo(result, 0)


def gen_hiss(freq, sr=SR):
    """Noise-like fricative sounds."""
    dur = DURATIONS['hiss']
    n = int(sr * dur)

    sig = noise(dur, sr)

    # Different fricative character based on frequency
    center = 3000 + (freq / 440) * 2000
    sig = bandpass(sig, max(20, center - 1000), min(sr/2 - 100, center + 1500), sr)

    env = env_adsr(dur, 0.01, 0.05, 0.6, 0.1, sr)
    sig *= env[:len(sig)] * 0.7

    result = normalize(sig, 0.85)
    fade_in(result, 3, sr)
    fade_out(result, 20, sr)
    return to_stereo(result, 0)


def gen_nasal(freq, sr=SR):
    """Warm humming nasal sounds."""
    dur = DURATIONS['nasal']
    n = int(sr * dur)

    source = glottal_source(freq, dur, sr)
    sig = lowpass(source, 500, sr)

    # Nasal resonance
    nasal = bandpass(source, max(20, 200), min(sr/2-100, 400), sr) * 0.5
    sig = sig[:n] + nasal[:n]

    # Anti-formant
    notch = bandpass(source, max(20, 250), min(sr/2-100, 350), sr) * 0.3
    sig -= notch[:n]

    env = env_adsr(dur, 0.05, 0.1, 0.8, 0.1, sr)
    sig *= env[:len(sig)]

    result = normalize(sig, 0.85)
    fade_in(result, 5, sr)
    fade_out(result, 30, sr)
    return to_stereo(result, 0)


def gen_liquid(freq, sr=SR):
    """Flowing l/r sounds — between tonal and noise."""
    dur = DURATIONS['liquid']
    n = int(sr * dur)
    t = np.arange(n) / sr

    source = glottal_source(freq, dur, sr)
    n_source = noise(dur, sr) * 0.3

    # Formant with modulation
    f1 = 400 + 200 * np.sin(2 * np.pi * 3 * t[0])
    sig = formant_synth(source + n_source[:len(source)],
                        [(int(f1), 100, 0), (1200, 150, -6), (2400, 200, -12)], sr)

    env = env_adsr(dur, 0.03, 0.1, 0.6, 0.1, sr)
    sig *= env[:len(sig)]

    result = normalize(sig[:n], 0.85)
    fade_in(result, 5, sr)
    fade_out(result, 25, sr)
    return to_stereo(result, 0)


def gen_syllables(freq, sr=SR):
    """Full syllables for melody grid — consonant + vowel."""
    dur = DURATIONS['syllables']
    n = int(sr * dur)

    # Consonant burst
    burst = noise(0.015, sr)
    burst = highpass(burst, 500, sr) * 0.5
    burst_env = env_exp_decay(0.015, 0.5, 0.005, sr)
    burst *= burst_env

    # Vowel body
    source = glottal_source(freq, dur, sr)
    vowels = [
        [(730, 90, 0), (1090, 110, -6), (2440, 170, -12)],  # ba
        [(300, 60, 0), (870, 100, -6), (2240, 170, -12)],   # du
        [(270, 60, 0), (2290, 200, -6), (3010, 200, -12)],  # li
        [(530, 80, 0), (1840, 150, -6), (2480, 180, -12)],  # me
    ]
    vidx = int(freq) % len(vowels)
    vowel = formant_synth(source, vowels[vidx], sr)

    sig = np.zeros(n)
    sig[:len(burst)] += burst
    vowel_start = int(sr * 0.02)
    end = min(vowel_start + len(vowel), n)
    sig[vowel_start:end] += vowel[:end - vowel_start]

    env = env_adsr(dur, 0.02, 0.05, 0.7, 0.15, sr)
    sig *= env[:len(sig)]

    result = normalize(sig, 0.85)
    fade_in(result, 3, sr)
    fade_out(result, 30, sr)
    return to_stereo(result, 0)


INSTRUMENTS = {
    'vowels': gen_vowels, 'clicks': gen_clicks, 'hiss': gen_hiss,
    'nasal': gen_nasal, 'liquid': gen_liquid, 'syllables': gen_syllables,
}
MEL_INSTS = {'syllables'}

FX_CONFIG = {
    'vowels': {'delay': 0.15, 'reverb': 0.25, 'gain': 0.24},
    'clicks': {'delay': 0.08, 'reverb': 0.12, 'gain': 0.30},
    'hiss': {'delay': 0.20, 'reverb': 0.30, 'gain': 0.20},
    'nasal': {'delay': 0.12, 'reverb': 0.20, 'gain': 0.26},
    'liquid': {'delay': 0.18, 'reverb': 0.28, 'gain': 0.22},
    'syllables': {'delay': 0.15, 'reverb': 0.25, 'gain': 0.22},
}


def generate(out_dir, sr=SR, fmt='ogg'):
    manifest_insts = []

    for inst_id, gen_fn in INSTRUMENTS.items():
        freqs = MEL_FREQS if inst_id in MEL_INSTS else FREQS
        is_mel = inst_id in MEL_INSTS

        print(f'  {inst_id}: {len(freqs)} pitches')
        manifest_insts.append({
            'id': inst_id, 'label': inst_id,
            'pitchCount': len(freqs),
            'type': 'melody' if is_mel else 'main',
        })

        for i, freq in enumerate(freqs):
            sig = gen_fn(freq, sr)
            sig = normalize(sig, 0.85)
            path = os.path.join(out_dir, f'{inst_id}_{i}.{fmt}')
            export_ogg(sig, path, sr)

    ir = generate_reverb_ir(2.0, dark=0.5, sr=sr)
    export_ogg(ir, os.path.join(out_dir, 'reverb_ir.ogg'), sr)
    write_manifest(out_dir, manifest_insts, FX_CONFIG)
