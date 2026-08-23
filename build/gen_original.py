import os
import numpy as np
import music
from common import (
    SR, sine, noise, env_exp_decay, env_adsr,
    lowpass, highpass, bandpass, comb_filter,
    asymmetric_saturate, normalize, fade_in, fade_out,
    to_stereo, mix_stereo, export_ogg, write_sprite, write_manifest, generate_reverb_ir,
)

# Eight rows, top-down: C5 A4 G4 E4 D4 C4 A3 G3.  The device this mode
# ports used an untempered set that included a near-unison (243 and 239 Hz,
# 29 cents apart), so two rows were effectively the same note beating
# against itself.
ROOT, SCALE, TOP = 'A', 'minor_pentatonic', 'C5'
FREQS = music.descending_hz(ROOT, SCALE, TOP, 8)

COL_SEQS = [
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
]

STEP_VELS = [v / 127 for v in
             [127, 36, 64, 36, 127, 36, 64, 36, 127, 36, 72, 36, 127, 36, 90, 36]]

PATTERN_CELLS = [
    {'0-1': 1.0},
    {'0-1': 1.0, '1-1': 1.0},
    {'0-1': 1.0},
    {'0-1': 1.0, '1-2': 1.0},
    {'0-1': 1.0},
    {'0-1': 1.0, '1-3': 1.0},
    {'0-1': 1.0, '2-3': 1.0},
    {'0-1': 1.0, '3-3': 1.0},
    {'0-1': 1.0, '4-3': 1.0},
    {'0-1': 1.0, '4-3': 1.0, '2-5': 1.0},
    {'0-1': 1.0, '4-3': 1.0, '1-5': 1.0},
    {'0-1': 1.0, '4-3': 1.0, '3-5': 1.0},
    {'0-1': 1.0, '4-3': 1.0, '2-5': 1.0},
    {'0-1': 1.0, '4-3': 1.0, '3-5': 1.0},
    {'0-1': 1.0, '4-3': 1.0, '3-5': 1.0, '6-5': 1.0},
    {'0-1': 1.0, '4-3': 1.0, '3-5': 1.0, '7-5': 1.0},
    {'0-1': 1.0, '4-3': 1.0, '3-5': 1.0, '6-5': 1.0},
    {'0-1': 1.0, '4-3': 1.0, '3-5': 1.0, '7-5': 1.0},
    {'0-1': 1.0, '4-3': 1.0, '1-7': 1.0, '7-5': 1.0},
    {'0-1': 1.0, '4-3': 1.0, '3-5': 1.0, '1-7': 1.0, '7-5': 1.0},
    {'0-1': 1.0, '4-3': 1.0, '3-5': 1.0, '1-7': 1.0, '7-5': 1.0, '2-5': 1.0},
    {'0-1': 1.0, '4-3': 1.0, '3-5': 1.0, '1-7': 1.0, '7-5': 1.0},
    {'0-1': 1.0, '4-3': 1.0, '3-5': 1.0, '1-7': 1.0, '7-5': 1.0, '2-5': 1.0},
    {'0-1': 1.0, '4-3': 1.0, '3-5': 1.0, '1-7': 1.0, '7-5': 1.0},
    {'0-1': 1.0, '4-3': 1.0, '3-5': 1.0, '1-7': 1.0, '7-5': 1.0, '2-5': 1.0},
    {'0-1': 1.0, '1-7': 1.0, '2-10': 1.0, '4-3': 1.0, '7-5': 1.0},
]

CONFIG = {
    'id': 'original',
    'label': 'original',
    'mainGrids': [
        {'rows': 8, 'cols': 16, 'defaultInstrument': 'tone'},
    ],
    'mainGridLayout': {'cols': 1},
    'melodyGrids': None,
    'effects': {
        'reverbWet': 0.08,
        'reverbDark': 0.35,
        'reverbLength': 1.0,
        'delayL': 0.33,
        'delayR': 0.22,
        'delayFeedback': 0.25,
        'delayDarkLP': 3000,
        'delayWet': 0.15,
        'compThreshold': -18,
        'compRatio': 4,
    },
    'fx': {'tone': {'delay': 0.15, 'reverb': 0.25, 'gain': 1.0}},
    'defaultStepMs': 74,
    'numPatterns': len(PATTERN_CELLS),
    'cellSize': 28,
    'seqLen': 16,
    'colSeqs': COL_SEQS,
    'stepVels': STEP_VELS,
    'hideControls': ['melody'],
    'defaultPatterns': [
        {'grids': [{'cells': cells, 'instrument': 'tone'}],
         'melody': {'cells': {}, 'instrument': 'tone'}}
        for cells in PATTERN_CELLS
    ],
}

ENVELOPE = {
    'attack_ms': 0.11,
    'decay_time': 0.259,
    'harmonic_ratios': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0],
    'harmonic_amplitudes_db': [0, -3.3, -0.1, -11.5, -17.9, -11.0, -29.8, -16.8, -30.0, -40.8],
    'inharmonicity_cents': [0, 6, -2, 10, -4, 5, -7, 8, -3, 6],
}


def gen_tone(freq, env_profile, sr=SR):
    dur = 0.9
    n = int(sr * dur)
    t = np.arange(n) / sr

    atk_dur = 0.008
    atk_n = int(sr * atk_dur)
    atk = noise(atk_dur, sr)
    atk = bandpass(atk, max(20, freq * 0.5), min(sr / 2 - 100, freq * 4), sr)
    atk_env = env_exp_decay(atk_dur, 0.2, 0.003, sr)
    atk *= atk_env * 0.125

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

    body_exc = np.zeros(n)
    body_exc[:atk_n] = atk[:min(atk_n, len(atk))]
    delay = max(1, int(sr / freq))
    body = comb_filter(body_exc, delay, feedback=0.085, lp_freq=min(freq * 3, sr / 2 - 100), sr=sr)
    body *= 0.138

    sig = sig + body[:n]

    env = env_exp_decay(dur, env_profile['attack_ms'], env_profile['decay_time'], sr)
    sig *= env

    sig[:len(atk)] += atk[:min(len(atk), n)]

    sig = asymmetric_saturate(sig, drive=1.37, asymmetry=0.080)

    sig_hp = highpass(sig, 1500, sr) * 2.78
    sig = sig + sig_hp

    sig = normalize(sig, 0.85)
    fade_in(sig, env_profile['attack_ms'], sr)
    fade_out(sig, 40, sr)

    return to_stereo(sig, 0)


def generate(out_dir, sr=SR, fmt='ogg'):
    freqs = FREQS
    print(f'  tuning: {SCALE} on {ROOT} -- {music.spell(ROOT, SCALE, TOP, len(freqs))}')
    print(f'  tone: {len(freqs)} pitches')
    segments = [normalize(gen_tone(freq, ENVELOPE, sr), 0.85) for freq in freqs]

    entry = {'id': 'tone', 'label': 'tone', 'pitchCount': len(freqs), 'type': 'main'}
    entry.update(write_sprite(out_dir, 'tone', segments, sr, fmt))

    ir = generate_reverb_ir(2.0, dark=0.5, sr=sr)
    export_ogg(ir, os.path.join(out_dir, f'reverb_ir.{fmt}'), sr)

    config = dict(CONFIG)
    config['mainFreqs'] = list(freqs)
    write_manifest(out_dir, config, [entry], fmt=fmt)
