"""Generate voices mode audio assets.

Vocal synthesis: glottal pulse model + formant filters.
Clean choral sound inspired by Roomful of Teeth — pure intonation,
gentle vibrato, ensemble warmth through detuned unisons.
"""

import os
import numpy as np
from common import (
    SR, sine, noise,
    env_adsr, env_exp_decay, lowpass, highpass, bandpass,
    normalize, fade_in, fade_out,
    to_stereo, mix_stereo, export_ogg, write_manifest, generate_reverb_ir,
)

FREQS = [262, 220, 196, 165, 131, 110, 98, 82, 73, 65, 55, 49, 41, 33, 27, 21]
MEL_FREQS = [523, 440, 392, 330, 262, 220, 196, 165]

DURATIONS = {
    'throat': 3.0, 'overtone': 3.0, 'breath': 2.5, 'belt': 2.5, 'hum': 3.0,
    'aah': 1.2, 'ooh': 1.2, 'mmm': 1.2,
}

# Formant tables: (freq, bandwidth, gain_db) — based on vocal acoustics research
FORMANTS = {
    'a': [(730, 90, 0), (1090, 110, -6), (2440, 170, -14), (3400, 250, -20)],
    'i': [(270, 60, 0), (2290, 200, -6), (3010, 200, -14), (3300, 250, -20)],
    'u': [(300, 60, 0), (870, 100, -6), (2240, 170, -14), (3200, 250, -20)],
    'e': [(530, 80, 0), (1840, 150, -6), (2480, 180, -14), (3320, 250, -20)],
    'o': [(570, 80, 0), (840, 100, -6), (2410, 170, -14), (3400, 250, -20)],
}


def glottal_pulse(f0, dur, sr=SR, open_quotient=0.6):
    """Rosenberg glottal pulse model with gentle jitter and shimmer."""
    n = int(sr * dur)

    # Gentle jitter (pitch perturbation — keep subtle for clean sound)
    jitter = np.cumsum(np.random.normal(0, f0 * 0.002, n)) / sr
    phase = np.cumsum((f0 + jitter * f0 * 0.15) / sr)
    cycle_pos = phase % 1

    # Rosenberg waveform
    oq = open_quotient
    pulse = np.where(
        cycle_pos < oq,
        3 * (cycle_pos / oq) ** 2 - 2 * (cycle_pos / oq) ** 3,
        0
    )

    # Gentle shimmer
    shimmer = 1 + np.random.normal(0, 0.015, n)
    pulse *= shimmer

    # Spectral tilt — gentle differentiation
    pulse = np.diff(pulse, prepend=0)
    return pulse


def formant_filter(sig, formants, sr=SR):
    """Apply parallel formant filter bank."""
    out = np.zeros_like(sig)
    for f, bw, gain in formants:
        low = max(20, f - bw / 2)
        high = min(sr / 2 - 100, f + bw / 2)
        if low >= high:
            continue
        filtered = bandpass(sig, low, high, sr, order=2)
        out += filtered * (10 ** (gain / 20))
    return out


def add_vibrato(sig, f0, rate=5.0, depth_cents=30, sr=SR):
    """Apply gentle vibrato by resampling."""
    n = len(sig)
    t = np.arange(n) / sr
    # Delayed onset vibrato — ramps in over first 0.3s
    onset = np.clip(t / 0.3, 0, 1)
    mod = depth_cents / 1200 * np.sin(2 * np.pi * rate * t) * onset
    indices = np.arange(n) + mod * sr / max(f0, 20)
    indices = np.clip(indices, 0, n - 1).astype(int)
    return sig[indices]


def add_breathiness(sig, amount=0.06, sr=SR):
    """Mix gentle filtered noise for breathiness."""
    n = len(sig)
    breath = noise(n / sr, sr)
    breath = bandpass(breath, 800, min(5000, sr / 2 - 100), sr) * amount
    return sig + breath[:n]


def ensemble_detune(source_fn, freq, n_voices=3, spread_cents=8, pan_spread=0.4, sr=SR):
    """Create ensemble warmth by layering slightly detuned voices."""
    dur = len(source_fn(freq, sr)) / sr if callable(source_fn) else None
    voices = []
    for i in range(n_voices):
        detune = (i - (n_voices - 1) / 2) * spread_cents
        f = freq * (2 ** (detune / 1200))
        pan = (i - (n_voices - 1) / 2) / max(1, (n_voices - 1) / 2) * pan_spread
        sig = source_fn(f, sr)
        voices.append(to_stereo(sig / n_voices, pan))
    n = min(v.shape[0] for v in voices)
    result = np.zeros((n, 2))
    for v in voices:
        result += v[:n]
    return result


# ======================== INSTRUMENTS ========================

def _make_throat_source(freq, sr):
    dur = DURATIONS['throat']
    return glottal_pulse(freq, dur, sr, open_quotient=0.5)


def gen_throat(freq, sr=SR):
    """Deep throat singing — warm, dark, with subtle subharmonic."""
    dur = DURATIONS['throat']
    n = int(sr * dur)

    source = glottal_pulse(freq, dur, sr, open_quotient=0.5)

    # Dark formants — low and warm
    sig = formant_filter(source, [(200, 80, 0), (600, 100, -4), (1200, 120, -12)])
    sig = lowpass(sig, 1200, sr)

    # Gentle subharmonic blend (sine, not square)
    t = np.arange(n) / sr
    sub = sine(freq * 0.5, dur, sr) * 0.15
    sig = sig[:n] + sub[:n]

    sig = add_vibrato(sig, freq, rate=4.5, depth_cents=20, sr=sr)
    sig = add_breathiness(sig, 0.04, sr)
    env = env_adsr(dur, 0.15, 0.3, 0.7, 0.5, sr)
    sig *= env[:len(sig)]

    return normalize(to_stereo(sig, 0), 0.85)


def gen_overtone(freq, sr=SR):
    """Overtone singing — clear harmonic isolation with warm base."""
    dur = DURATIONS['overtone']
    n = int(sr * dur)

    source = glottal_pulse(freq, dur, sr, open_quotient=0.6)

    # Wide formant for the base vowel
    base = formant_filter(source, FORMANTS['o']) * 0.3

    # Narrow resonance at selected harmonic — varies by pitch
    harmonic = 3 if freq > 150 else 4 if freq > 80 else 5
    h_freq = freq * harmonic
    if h_freq < sr / 2 - 200:
        overtone = bandpass(source, max(20, h_freq - 50), min(sr/2 - 100, h_freq + 50), sr)
    else:
        overtone = np.zeros(len(source))

    sig = (base[:n] + overtone[:n] * 0.7)

    sig = add_vibrato(sig, freq, rate=4.0, depth_cents=15, sr=sr)
    sig = add_breathiness(sig, 0.06, sr)
    env = env_adsr(dur, 0.2, 0.3, 0.6, 0.6, sr)
    sig *= env[:len(sig)]

    return normalize(to_stereo(sig, 0), 0.85)


def gen_breath(freq, sr=SR):
    """Breathy whisper-singing — mostly air with pitched hint."""
    dur = DURATIONS['breath']
    n = int(sr * dur)

    source_noise = noise(dur, sr)
    source_glottal = glottal_pulse(freq, dur, sr, open_quotient=0.7)
    source = source_noise * 0.75 + source_glottal[:len(source_noise)] * 0.25

    sig = formant_filter(source, FORMANTS['a'])
    sig = lowpass(sig, 6000, sr)

    env = env_adsr(dur, 0.3, 0.3, 0.5, 0.6, sr)
    sig *= env[:len(sig)] * 0.5

    return normalize(to_stereo(sig, 0), 0.85)


def gen_belt(freq, sr=SR):
    """Full-voice belting — powerful but clean, like a trained singer."""
    dur = DURATIONS['belt']
    n = int(sr * dur)

    # Higher open quotient for brighter, fuller sound
    source = glottal_pulse(freq, dur, sr, open_quotient=0.75)

    # Bright, open formants
    sig = formant_filter(source, [
        (800, 120, 0), (1200, 130, -3), (2800, 200, -10), (3500, 250, -16)
    ])

    # No saturation — clean power through amplitude and formant placement
    sig = add_vibrato(sig[:n], freq, rate=5.5, depth_cents=35, sr=sr)
    sig = add_breathiness(sig, 0.04, sr)

    env = env_adsr(dur, 0.06, 0.12, 0.85, 0.35, sr)
    sig *= env[:len(sig)]

    return normalize(to_stereo(sig, 0), 0.85)


def gen_hum(freq, sr=SR):
    """Closed-mouth humming — warm and nasal."""
    dur = DURATIONS['hum']
    n = int(sr * dur)

    source = glottal_pulse(freq, dur, sr, open_quotient=0.5)

    # Very low-pass — closed mouth resonance
    sig = lowpass(source, 600, sr)
    # Nasal resonance around 250-300 Hz
    nasal = bandpass(source, 200, 350, sr) * 0.3
    sig = sig[:n] + nasal[:n]

    sig = add_vibrato(sig, freq, rate=4.5, depth_cents=20, sr=sr)
    sig = add_breathiness(sig, 0.02, sr)
    env = env_adsr(dur, 0.25, 0.3, 0.8, 0.6, sr)
    sig *= env[:len(sig)]

    return normalize(to_stereo(sig, 0), 0.85)


def gen_aah(freq, sr=SR):
    """Open 'aah' vowel — clean, clear single voice with gentle onset."""
    dur = DURATIONS['aah']
    n = int(sr * dur)

    source = glottal_pulse(freq, dur, sr, open_quotient=0.6)
    sig = formant_filter(source, FORMANTS['a'])
    sig = sig[:n]
    sig = add_vibrato(sig, freq, rate=5, depth_cents=12, sr=sr)
    sig = add_breathiness(sig, 0.02, sr)
    env = env_adsr(dur, 0.05, 0.15, 0.65, 0.3, sr)
    sig *= env[:len(sig)]

    return normalize(to_stereo(sig, 0), 0.85)


def gen_ooh(freq, sr=SR):
    """Round 'ooh' vowel — warm, focused."""
    dur = DURATIONS['ooh']
    n = int(sr * dur)

    source = glottal_pulse(freq, dur, sr, open_quotient=0.55)
    sig = formant_filter(source, FORMANTS['u'])
    sig = sig[:n]
    sig = add_vibrato(sig, freq, rate=5, depth_cents=10, sr=sr)
    sig = add_breathiness(sig, 0.02, sr)
    env = env_adsr(dur, 0.05, 0.15, 0.65, 0.3, sr)
    sig *= env[:len(sig)]

    return normalize(to_stereo(sig, 0), 0.85)


def gen_mmm(freq, sr=SR):
    """Closed 'mmm' — gentle hum, clean."""
    dur = DURATIONS['mmm']
    n = int(sr * dur)

    source = glottal_pulse(freq, dur, sr, open_quotient=0.5)
    sig = lowpass(source, 500, sr)
    nasal = bandpass(source, 230, 330, sr) * 0.25
    sig = sig[:n] + nasal[:n]
    sig = sig[:n]
    sig = add_vibrato(sig, freq, rate=4.5, depth_cents=10, sr=sr)
    env = env_adsr(dur, 0.05, 0.15, 0.7, 0.3, sr)
    sig *= env[:len(sig)]

    return normalize(to_stereo(sig, 0), 0.85)


INSTRUMENTS = {
    'throat': gen_throat, 'overtone': gen_overtone, 'breath': gen_breath,
    'belt': gen_belt, 'hum': gen_hum,
    'aah': gen_aah, 'ooh': gen_ooh, 'mmm': gen_mmm,
}
MEL_INSTS = {'aah', 'ooh', 'mmm'}

FX_CONFIG = {
    'throat': {'delay': 0.12, 'reverb': 0.30, 'gain': 0.24},
    'overtone': {'delay': 0.15, 'reverb': 0.35, 'gain': 0.20},
    'breath': {'delay': 0.20, 'reverb': 0.45, 'gain': 0.16},
    'belt': {'delay': 0.10, 'reverb': 0.25, 'gain': 0.26},
    'hum': {'delay': 0.15, 'reverb': 0.30, 'gain': 0.22},
    'aah': {'delay': 0.08, 'reverb': 0.15, 'gain': 0.24},
    'ooh': {'delay': 0.08, 'reverb': 0.15, 'gain': 0.22},
    'mmm': {'delay': 0.06, 'reverb': 0.12, 'gain': 0.24},
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

    ir = generate_reverb_ir(3.5, dark=0.65, sr=sr)
    export_ogg(ir, os.path.join(out_dir, 'reverb_ir.ogg'), sr)
    write_manifest(out_dir, manifest_insts, FX_CONFIG)
