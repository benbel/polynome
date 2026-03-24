"""Generate voices mode audio assets.

Vocal synthesis: glottal pulse model + formant filters.
Choir discovering music — long sustained vowels, human vibrato,
overtone singing, percussive clicks for contrast.
"""

import os
import numpy as np
from common import (
    SR, sine, noise,
    env_adsr, env_exp_decay, lowpass, highpass, bandpass,
    normalize, fade_in, fade_out,
    to_stereo, mix_stereo, export_ogg, write_manifest, generate_reverb_ir,
)

FREQS = [523, 440, 370, 311, 262, 220, 175, 147, 123, 104, 82, 65, 55, 44, 33, 25]
MEL_FREQS = [659, 523, 440, 349, 262, 220, 175, 131]

DURATIONS = {
    'throat': 3.0, 'overtone': 3.0, 'breath': 2.5, 'belt': 2.5, 'click': 0.8,
    'aah': 2.8, 'ooh': 2.8, 'mmm': 2.8,
}

# Formant tables: (freq, bandwidth, gain_db)
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
    jitter = np.cumsum(np.random.normal(0, f0 * 0.002, n)) / sr
    phase = np.cumsum((f0 + jitter * f0 * 0.15) / sr)
    cycle_pos = phase % 1
    oq = open_quotient
    pulse = np.where(
        cycle_pos < oq,
        3 * (cycle_pos / oq) ** 2 - 2 * (cycle_pos / oq) ** 3,
        0
    )
    shimmer = 1 + np.random.normal(0, 0.015, n)
    pulse *= shimmer
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


def pitch_track_formants(base_formants, freq):
    """Adjust formants for low frequencies to avoid unnatural timbre.

    When fundamental is low, formants need to shift down slightly
    to maintain natural vocal character.
    """
    if freq >= 150:
        return base_formants
    # Scale factor: at 25Hz, shift formants down ~30%
    scale = 0.7 + 0.3 * min(1, freq / 150)
    return [(f * scale, bw * scale, g) for f, bw, g in base_formants]


def add_vibrato(sig, f0, rate=5.0, depth_cents=30, sr=SR):
    """Apply gentle vibrato by resampling."""
    n = len(sig)
    t = np.arange(n) / sr
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


# ======================== INSTRUMENTS ========================

def gen_throat(freq, sr=SR):
    """Deep throat singing — warm, dark, with subtle subharmonic."""
    dur = DURATIONS['throat']
    n = int(sr * dur)

    source = glottal_pulse(freq, dur, sr, open_quotient=0.5)

    # Dark formants with pitch tracking
    formants = pitch_track_formants(
        [(200, 80, 0), (600, 100, -4), (1200, 120, -12)], freq)
    sig = formant_filter(source, formants)
    sig = lowpass(sig, max(200, min(1200, freq * 4)), sr)

    sub = sine(max(20, freq * 0.5), dur, sr) * 0.15
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

    base_formants = pitch_track_formants(FORMANTS['o'], freq)
    base = formant_filter(source, base_formants) * 0.3

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
    """Breathy whisper-singing — 90% air with pitched hint."""
    dur = DURATIONS['breath']
    n = int(sr * dur)

    source_noise = noise(dur, sr)
    source_glottal = glottal_pulse(freq, dur, sr, open_quotient=0.7)
    # 90% noise, 10% glottal — mostly air with just a hint of pitch
    source = source_noise * 0.9 + source_glottal[:len(source_noise)] * 0.1

    formants = pitch_track_formants(FORMANTS['a'], freq)
    sig = formant_filter(source, formants)
    sig = lowpass(sig, 6000, sr)

    env = env_adsr(dur, 0.3, 0.3, 0.5, 0.6, sr)
    sig *= env[:len(sig)] * 0.5

    return normalize(to_stereo(sig, 0), 0.85)


def gen_belt(freq, sr=SR):
    """Full-voice belting — powerful, clean, trained singer. Tighter vibrato."""
    dur = DURATIONS['belt']
    n = int(sr * dur)

    source = glottal_pulse(freq, dur, sr, open_quotient=0.75)

    formants = pitch_track_formants(
        [(800, 120, 0), (1200, 130, -3), (2800, 200, -10), (3500, 250, -16)], freq)
    sig = formant_filter(source, formants)

    # Tighter vibrato (25 cents instead of 35)
    sig = add_vibrato(sig[:n], freq, rate=5.5, depth_cents=25, sr=sr)
    sig = add_breathiness(sig, 0.04, sr)

    env = env_adsr(dur, 0.06, 0.12, 0.85, 0.35, sr)
    sig *= env[:len(sig)]

    return normalize(to_stereo(sig, 0), 0.85)


def gen_click(freq, sr=SR):
    """Percussive vocal clicks — tongue clicks, plosive bursts for rhythmic contrast."""
    dur = DURATIONS['click']
    n = int(sr * dur)
    sig = np.zeros(n)

    # Click type varies by pitch register
    # Low pitches: deep tongue click, high: sharp alveolar click
    if freq > 300:
        # Sharp palatal click
        burst = noise(0.003, sr)
        burst = bandpass(burst, 2000, min(sr / 2 - 100, 8000), sr) * 0.8
        sig[:len(burst)] = burst[:min(len(burst), n)]
    elif freq > 100:
        # Lateral tongue click
        burst = noise(0.005, sr)
        burst = bandpass(burst, 800, min(sr / 2 - 100, 4000), sr) * 0.7
        sig[:len(burst)] = burst[:min(len(burst), n)]
        # Resonant tail
        tail = sine(freq, 0.04, sr) * 0.15
        tail_env = env_exp_decay(0.04, 0.3, 0.015, sr)
        tail *= tail_env
        start = len(burst)
        end = min(start + len(tail), n)
        sig[start:end] += tail[:end - start]
    else:
        # Deep glottal pop
        burst = noise(0.008, sr)
        burst = lowpass(burst, min(1500, sr / 2 - 100), sr) * 0.6
        sig[:len(burst)] = burst[:min(len(burst), n)]
        # Sub thump
        thump = sine(max(20, freq), 0.06, sr) * 0.2
        thump_env = env_exp_decay(0.06, 0.5, 0.02, sr)
        thump *= thump_env
        end = min(len(burst) + len(thump), n)
        sig[len(burst):end] += thump[:end - len(burst)]

    # Light room resonance
    env = env_exp_decay(dur, 0.3, 0.08, sr)
    sig *= env[:n]

    return normalize(to_stereo(sig, 0), 0.85)


def gen_aah(freq, sr=SR):
    """Open 'aah' vowel — sustained, gentle vibrato, pitch-tracking formants."""
    dur = DURATIONS['aah']
    n = int(sr * dur)

    source = glottal_pulse(freq, dur, sr, open_quotient=0.6)
    formants = pitch_track_formants(FORMANTS['a'], freq)
    sig = formant_filter(source, formants)
    sig = sig[:n]
    sig = add_vibrato(sig, freq, rate=5, depth_cents=12, sr=sr)
    sig = add_breathiness(sig, 0.02, sr)
    env = env_adsr(dur, 0.12, 0.2, 0.7, 0.6, sr)
    sig *= env[:len(sig)]

    return normalize(to_stereo(sig, 0), 0.85)


def gen_ooh(freq, sr=SR):
    """Round 'ooh' vowel — sustained, warm, pitch-tracking formants."""
    dur = DURATIONS['ooh']
    n = int(sr * dur)

    source = glottal_pulse(freq, dur, sr, open_quotient=0.55)
    formants = pitch_track_formants(FORMANTS['u'], freq)
    sig = formant_filter(source, formants)
    sig = sig[:n]
    sig = add_vibrato(sig, freq, rate=5, depth_cents=10, sr=sr)
    sig = add_breathiness(sig, 0.02, sr)
    env = env_adsr(dur, 0.12, 0.2, 0.7, 0.6, sr)
    sig *= env[:len(sig)]

    return normalize(to_stereo(sig, 0), 0.85)


def gen_mmm(freq, sr=SR):
    """Closed 'mmm' — sustained gentle hum, pitch-tracking."""
    dur = DURATIONS['mmm']
    n = int(sr * dur)

    source = glottal_pulse(freq, dur, sr, open_quotient=0.5)
    lp_freq = max(200, min(500, freq * 3))
    sig = lowpass(source, lp_freq, sr)
    nasal_center = max(50, min(330, freq * 1.5))
    nasal = bandpass(source,
                     max(20, nasal_center - 50),
                     min(sr / 2 - 100, nasal_center + 50), sr) * 0.25
    sig = sig[:n] + nasal[:n]
    sig = sig[:n]
    sig = add_vibrato(sig, freq, rate=4.5, depth_cents=10, sr=sr)
    env = env_adsr(dur, 0.12, 0.2, 0.75, 0.6, sr)
    sig *= env[:len(sig)]

    return normalize(to_stereo(sig, 0), 0.85)


INSTRUMENTS = {
    'throat': gen_throat, 'overtone': gen_overtone, 'breath': gen_breath,
    'belt': gen_belt, 'click': gen_click,
    'aah': gen_aah, 'ooh': gen_ooh, 'mmm': gen_mmm,
}
MEL_INSTS = {'aah', 'ooh', 'mmm'}

FX_CONFIG = {
    'throat': {'delay': 0.12, 'reverb': 0.30, 'gain': 0.24},
    'overtone': {'delay': 0.15, 'reverb': 0.35, 'gain': 0.20},
    'breath': {'delay': 0.20, 'reverb': 0.45, 'gain': 0.16},
    'belt': {'delay': 0.10, 'reverb': 0.25, 'gain': 0.26},
    'click': {'delay': 0.06, 'reverb': 0.08, 'gain': 0.30},
    'aah': {'delay': 0.10, 'reverb': 0.20, 'gain': 0.22},
    'ooh': {'delay': 0.10, 'reverb': 0.20, 'gain': 0.20},
    'mmm': {'delay': 0.08, 'reverb': 0.15, 'gain': 0.22},
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
