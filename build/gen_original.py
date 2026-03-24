"""Generate original mode audio assets.

Additive synthesis matching a monome grid video's timbral profile.
Uses analysis results if available, otherwise sensible defaults.
"""

import os, json
import numpy as np
from common import (
    SR, sine, env_exp_decay, asymmetric_saturate,
    normalize, fade_in, fade_out, to_stereo,
    export_ogg, write_manifest, generate_reverb_ir,
)

# Default scale if no analysis available
DEFAULT_FREQS = [440, 392, 330, 294, 262, 220, 196, 165]

# Brighter, more bell-like tone — joyful, sparkling character
DEFAULT_ENVELOPE = {
    'attack_ms': 2.0,
    'decay_time': 0.35,
    'harmonic_ratios': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
    'harmonic_amplitudes_db': [0, -5, -10, -14, -18, -22, -30],
    'inharmonicity_cents': [0, 8, -3, 12, -5, 6, -8],
}


def gen_tone(freq, env_profile, sr=SR):
    dur = 0.5
    n = int(sr * dur)
    t = np.arange(n) / sr

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

    # Envelope
    env = env_exp_decay(dur, env_profile['attack_ms'], env_profile['decay_time'], sr)
    sig *= env

    # Very gentle warmth — minimal saturation to keep brightness
    sig = asymmetric_saturate(sig, drive=1.05, asymmetry=0.02)

    sig = normalize(sig, 0.85)
    fade_in(sig, env_profile['attack_ms'], sr)
    fade_out(sig, 20, sr)

    return to_stereo(sig, 0)


def generate(out_dir, sr=SR, fmt='ogg'):
    # Try loading analysis results
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

    # Reverb IR
    ir = generate_reverb_ir(2.0, dark=0.5, sr=sr)
    export_ogg(ir, os.path.join(out_dir, 'reverb_ir.ogg'), sr)

    write_manifest(out_dir, [
        {'id': 'tone', 'label': 'tone', 'pitchCount': len(freqs), 'type': 'main'}
    ], {
        'tone': {'delay': 0.15, 'reverb': 0.25, 'gain': 0.3}
    })
