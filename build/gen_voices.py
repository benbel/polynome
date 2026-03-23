"""Generate voices mode audio assets.

Vocal synthesis: glottal pulse model + formant filters.
Inspired by Roomful of Teeth, Meredith Monk.
"""

import os
import numpy as np
from common import (
    SR, sine, noise,
    env_adsr, env_exp_decay, lowpass, highpass, bandpass,
    tanh_saturate, normalize, fade_in, fade_out,
    to_stereo, export_ogg, write_manifest, generate_reverb_ir,
)

FREQS = [262, 220, 196, 165, 131, 110, 98, 82, 73, 65, 55, 49, 41, 33, 27, 21]
MEL_FREQS = [523, 440, 392, 330, 262, 220, 196, 165]

DURATIONS = {
    'throat': 2.5, 'overtone': 2.5, 'breath': 2.0, 'belt': 2.0, 'hum': 2.5,
    'aah': 3.0, 'ooh': 3.0, 'mmm': 3.0,
}

# Formant tables: (freq, bandwidth, gain_db)
FORMANTS = {
    'a': [(730, 90, 0), (1090, 110, -6), (2440, 170, -12), (3400, 250, -18)],
    'i': [(270, 60, 0), (2290, 200, -6), (3010, 200, -12), (3300, 250, -18)],
    'u': [(300, 60, 0), (870, 100, -6), (2240, 170, -12), (3200, 250, -18)],
    'e': [(530, 80, 0), (1840, 150, -6), (2480, 180, -12), (3320, 250, -18)],
    'o': [(570, 80, 0), (840, 100, -6), (2410, 170, -12), (3400, 250, -18)],
}


def glottal_pulse(f0, dur, sr=SR, open_quotient=0.6):
    """Rosenberg glottal pulse model with jitter and shimmer."""
    n = int(sr * dur)
    t = np.arange(n) / sr

    # Jitter (pitch perturbation)
    jitter = np.cumsum(np.random.normal(0, f0 * 0.005, n)) / sr
    phase = np.cumsum((f0 + jitter * f0 * 0.3) / sr)
    cycle_pos = phase % 1

    # Rosenberg waveform
    oq = open_quotient
    pulse = np.where(
        cycle_pos < oq,
        3 * (cycle_pos / oq) ** 2 - 2 * (cycle_pos / oq) ** 3,
        0
    )

    # Shimmer
    shimmer = 1 + np.random.normal(0, 0.03, n)
    pulse *= shimmer

    # Spectral tilt
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


def add_vibrato(sig, f0, rate=5.5, depth_cents=40, sr=SR):
    """Apply vibrato by resampling."""
    n = len(sig)
    t = np.arange(n) / sr
    mod = depth_cents / 1200 * np.sin(2 * np.pi * rate * t)
    # Simple phase modulation approximation
    indices = np.arange(n) + mod * sr / f0
    indices = np.clip(indices, 0, n - 1).astype(int)
    return sig[indices]


def add_breathiness(sig, amount=0.1, sr=SR):
    """Mix filtered noise for breathiness."""
    n = len(sig)
    breath = noise(n / sr, sr)
    breath = bandpass(breath, 500, min(4000, sr / 2 - 100), sr) * amount
    return sig + breath[:n]


# ======================== INSTRUMENTS ========================

def gen_throat(freq, sr=SR):
    dur = DURATIONS['throat']
    n = int(sr * dur)

    source = glottal_pulse(freq, dur, sr, open_quotient=0.5)

    # Very low F1, dark LP
    sig = formant_filter(source, [(200, 60, 0), (800, 100, -8)])
    sig = lowpass(sig, 800, sr)

    # Subharmonic via ring modulation with square at f0/2
    t = np.arange(n) / sr
    sub_sq = np.sign(np.sin(2 * np.pi * freq * 0.5 * t))
    sig = sig[:n] * (0.7 + 0.3 * sub_sq)

    sig = add_breathiness(sig, 0.05, sr)
    env = env_adsr(dur, 0.1, 0.2, 0.7, 0.4, sr)
    sig *= env[:n]

    return normalize(to_stereo(sig, 0), 0.85)


def gen_overtone(freq, sr=SR):
    dur = DURATIONS['overtone']
    n = int(sr * dur)

    source = glottal_pulse(freq, dur, sr, open_quotient=0.6)

    # Narrow BP at selected harmonic (we'll use 4th harmonic as default)
    harmonic = 4
    sig = bandpass(source, freq * harmonic - 30, min(sr/2 - 100, freq * harmonic + 30), sr)

    # Suppress fundamental
    sig_hp = highpass(sig, freq * 1.5, sr)
    sig = sig_hp * 0.7 + sig * 0.3

    sig = add_breathiness(sig[:n], 0.08, sr)
    env = env_adsr(dur, 0.15, 0.2, 0.6, 0.5, sr)
    sig *= env[:len(sig)]

    return normalize(to_stereo(sig, 0), 0.85)


def gen_breath(freq, sr=SR):
    dur = DURATIONS['breath']
    n = int(sr * dur)

    # 80% noise + 20% glottal
    source_noise = noise(dur, sr)
    source_glottal = glottal_pulse(freq, dur, sr, open_quotient=0.7)
    source = source_noise * 0.8 + source_glottal[:len(source_noise)] * 0.2

    sig = formant_filter(source, FORMANTS['a'])
    sig = highpass(sig, 500, sr)

    env = env_adsr(dur, 0.3, 0.3, 0.5, 0.5, sr)
    sig *= env[:len(sig)] * 0.6

    return normalize(to_stereo(sig, 0), 0.85)


def gen_belt(freq, sr=SR):
    dur = DURATIONS['belt']
    n = int(sr * dur)

    source = glottal_pulse(freq, dur, sr, open_quotient=0.75)

    # Bright formants
    sig = formant_filter(source, [(800, 100, 0), (1200, 120, -3), (2800, 200, -10)])

    # Slight saturation + vibrato
    sig = tanh_saturate(sig, 1.5)
    sig = add_vibrato(sig[:n], freq, rate=5.5, depth_cents=40, sr=sr)
    sig = add_breathiness(sig, 0.05, sr)

    env = env_adsr(dur, 0.05, 0.1, 0.8, 0.3, sr)
    sig *= env[:len(sig)]

    return normalize(to_stereo(sig, 0), 0.85)


def gen_hum(freq, sr=SR):
    dur = DURATIONS['hum']
    n = int(sr * dur)

    source = glottal_pulse(freq, dur, sr, open_quotient=0.5)

    # All formants LP'd below 500Hz, nasal anti-formant
    sig = lowpass(source, 500, sr)
    # Anti-formant at 250Hz (notch approximation)
    notch = bandpass(source, 200, 300, sr)
    sig = sig - notch * 0.5

    sig = add_breathiness(sig[:n], 0.03, sr)
    env = env_adsr(dur, 0.2, 0.2, 0.8, 0.5, sr)
    sig *= env[:len(sig)]

    return normalize(to_stereo(sig, 0), 0.85)


def gen_aah(freq, sr=SR):
    dur = DURATIONS['aah']
    source = glottal_pulse(freq, dur, sr)
    sig = formant_filter(source, FORMANTS['a'])
    sig = add_vibrato(sig, freq, rate=5, depth_cents=30, sr=sr)
    sig = add_breathiness(sig, 0.08, sr)
    env = env_adsr(dur, 0.3, 0.2, 0.7, 0.8, sr)
    sig *= env[:len(sig)]
    return normalize(to_stereo(sig, 0), 0.85)


def gen_ooh(freq, sr=SR):
    dur = DURATIONS['ooh']
    source = glottal_pulse(freq, dur, sr)
    sig = formant_filter(source, FORMANTS['u'])
    sig = add_vibrato(sig, freq, rate=5, depth_cents=25, sr=sr)
    sig = add_breathiness(sig, 0.06, sr)
    env = env_adsr(dur, 0.3, 0.2, 0.7, 0.8, sr)
    sig *= env[:len(sig)]
    return normalize(to_stereo(sig, 0), 0.85)


def gen_mmm(freq, sr=SR):
    dur = DURATIONS['mmm']
    source = glottal_pulse(freq, dur, sr)
    sig = lowpass(source, 400, sr)
    # Nasal anti-formant
    notch = bandpass(source, 250, 350, sr)
    sig = sig - notch * 0.4
    sig = add_vibrato(sig[:len(sig)], freq, rate=4.5, depth_cents=20, sr=sr)
    sig = add_breathiness(sig, 0.04, sr)
    env = env_adsr(dur, 0.3, 0.2, 0.8, 0.8, sr)
    sig *= env[:len(sig)]
    return normalize(to_stereo(sig, 0), 0.85)


INSTRUMENTS = {
    'throat': gen_throat, 'overtone': gen_overtone, 'breath': gen_breath,
    'belt': gen_belt, 'hum': gen_hum,
    'aah': gen_aah, 'ooh': gen_ooh, 'mmm': gen_mmm,
}
MEL_INSTS = {'aah', 'ooh', 'mmm'}

FX_CONFIG = {
    'throat': {'delay': 0.10, 'reverb': 0.20, 'gain': 0.28},
    'overtone': {'delay': 0.15, 'reverb': 0.30, 'gain': 0.22},
    'breath': {'delay': 0.20, 'reverb': 0.40, 'gain': 0.18},
    'belt': {'delay': 0.08, 'reverb': 0.15, 'gain': 0.30},
    'hum': {'delay': 0.12, 'reverb': 0.25, 'gain': 0.26},
    'aah': {'delay': 0.18, 'reverb': 0.35, 'gain': 0.20},
    'ooh': {'delay': 0.20, 'reverb': 0.38, 'gain': 0.18},
    'mmm': {'delay': 0.15, 'reverb': 0.30, 'gain': 0.22},
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

    ir = generate_reverb_ir(3.0, dark=0.6, sr=sr)
    export_ogg(ir, os.path.join(out_dir, 'reverb_ir.ogg'), sr)
    write_manifest(out_dir, manifest_insts, FX_CONFIG)
