"""Generate original mode audio assets.

Warm, woody, marimba-like tone with natural decay.
Matches the monome Press Cafe timbral profile.
"""

import os, json
import numpy as np
from common import (
    SR, sine, noise, env_exp_decay, env_adsr,
    lowpass, highpass, bandpass, comb_filter,
    asymmetric_saturate, normalize, fade_in, fade_out,
    to_stereo, mix_stereo, export_ogg, write_manifest, generate_reverb_ir,
)

DEFAULT_FREQS = [587, 494, 415, 349, 294, 247, 196, 147]

# Warm, woody tone — marimba-like with body resonance
DEFAULT_ENVELOPE = {
    'attack_ms': 1.5,
    'decay_time': 0.45,
    'harmonic_ratios': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0],
    'harmonic_amplitudes_db': [0, -3, -6, -10, -14, -18, -22, -26, -32, -38],
    'inharmonicity_cents': [0, 6, -2, 10, -4, 5, -7, 8, -3, 6],
}


def gen_tone(freq, env_profile, sr=SR):
    dur = 0.9
    n = int(sr * dur)
    t = np.arange(n) / sr

    # --- Mallet attack transient ---
    # Short noise burst filtered around fundamental for woody thump
    atk_dur = 0.008
    atk_n = int(sr * atk_dur)
    atk = noise(atk_dur, sr)
    atk = bandpass(atk, max(20, freq * 0.5), min(sr / 2 - 100, freq * 4), sr)
    atk_env = env_exp_decay(atk_dur, 0.2, 0.003, sr)
    atk *= atk_env * 0.35

    # --- Tonal body: slightly detuned partials for warmth ---
    sig = np.zeros(n)
    ratios = env_profile['harmonic_ratios']
    amps_db = env_profile['harmonic_amplitudes_db']
    detune = env_profile.get('inharmonicity_cents', [0] * len(ratios))

    for h in range(len(ratios)):
        partial_freq = freq * ratios[h] * (2 ** (detune[h] / 1200))
        if partial_freq >= sr / 2:
            break
        amp = 10 ** (amps_db[h] / 20)
        sig += amp * np.sin(2 * np.pi * partial_freq * t)

    # --- Body resonance: low-mid bump via comb resonator ---
    body_exc = np.zeros(n)
    body_exc[:atk_n] = atk[:min(atk_n, len(atk))]
    delay = max(1, int(sr / freq))
    body = comb_filter(body_exc, delay, feedback=0.6, lp_freq=min(freq * 3, sr / 2 - 100), sr=sr)
    body *= 0.15

    # Combine tonal + body
    sig = sig + body[:n]

    # Envelope — longer natural decay
    env = env_exp_decay(dur, env_profile['attack_ms'], env_profile['decay_time'], sr)
    sig *= env

    # Insert attack transient
    sig[:len(atk)] += atk[:min(len(atk), n)]

    # Gentle warmth — minimal asymmetric saturation
    sig = asymmetric_saturate(sig, drive=1.20, asymmetry=0.05)

    # High-frequency boost to match reference brightness
    sig_hp = highpass(sig, 1500, sr) * 1.0
    sig = sig + sig_hp

    sig = normalize(sig, 0.85)
    fade_in(sig, env_profile['attack_ms'], sr)
    fade_out(sig, 40, sr)

    return to_stereo(sig, 0)


def generate(out_dir, sr=SR, fmt='ogg'):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    analysis_dir = os.path.join(root, 'analysis')

    freqs = DEFAULT_FREQS
    env_profile = DEFAULT_ENVELOPE

    scale_path = os.path.join(analysis_dir, 'scale_detected.json')
    if os.path.exists(scale_path):
        with open(scale_path) as f:
            scale = json.load(f)
        freqs = scale['pitches_hz']

    env_path = os.path.join(analysis_dir, 'envelope_profile.json')
    if os.path.exists(env_path):
        with open(env_path) as f:
            env_profile = json.load(f)

    print(f'  tone: {len(freqs)} pitches')
    for i, freq in enumerate(freqs):
        sig = gen_tone(freq, env_profile, sr)
        sig = normalize(sig, 0.85)
        path = os.path.join(out_dir, f'tone_{i}.{fmt}')
        export_ogg(sig, path, sr)

    ir = generate_reverb_ir(2.0, dark=0.5, sr=sr)
    export_ogg(ir, os.path.join(out_dir, 'reverb_ir.ogg'), sr)

    write_manifest(out_dir, [
        {'id': 'tone', 'label': 'tone', 'pitchCount': len(freqs), 'type': 'main'}
    ], {
        'tone': {'delay': 0.15, 'reverb': 0.25, 'gain': 0.3}
    })
