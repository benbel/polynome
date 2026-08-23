import os
import numpy as np
import music
from common import (
    SR, sine, noise, env_exp_decay, env_adsr,
    lowpass, highpass, bandpass, comb_filter, high_shelf, low_shelf,
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
        'compThreshold': -14,
        'compRatio': 4,
    },
    'fx': {'tone': {'delay': 0.15, 'reverb': 0.25, 'gain': 0.44}},
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

TONE_DUR = 0.7
PAN_SPREAD = 0.3

ENVELOPE = {
    'attack_ms': 1.2,
    'decay_time': 0.30,
    'partial_damping': 0.42,
    'harmonic_ratios': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0],
    'harmonic_amplitudes_db': [0, -3.3, -0.1, -11.5, -17.9, -11.0, -29.8, -16.8, -30.0, -40.8],
    'inharmonicity_cents': [0, 6, -2, 10, -4, 5, -7, 8, -3, 6],
    'partial_phases': [0.00, 2.41, 4.83, 1.17, 3.62, 5.94, 0.78, 2.05, 4.26, 3.11],
    'presence_db': 6.0,
    'warmth_db': 2.5,
}


def pan_positions(freqs, spread=PAN_SPREAD):
    order = sorted(range(len(freqs)), key=lambda i: freqs[i])
    pans = [0.0] * len(freqs)
    for rank, i in enumerate(order):
        pans[i] = -spread + 2 * spread * rank / max(1, len(freqs) - 1)
    return pans


def gen_tone(freq, env_profile, pan=0.0, sr=SR):
    dur = TONE_DUR
    n = int(sr * dur)
    t = np.arange(n) / sr

    ratios = env_profile['harmonic_ratios']
    amps_db = env_profile['harmonic_amplitudes_db']
    detune = env_profile.get('inharmonicity_cents', [0] * len(ratios))
    phases = env_profile.get('partial_phases', [0.0] * len(ratios))
    decay_time = env_profile['decay_time']
    damping = env_profile['partial_damping']

    sig = np.zeros(n)
    for h in range(len(ratios)):
        partial_freq = freq * ratios[h] * (2 ** (detune[h] / 1200))
        if partial_freq >= min(sr / 2, 16000):
            break
        amp = 10 ** (amps_db[h] / 20)
        partial_decay = decay_time * ratios[h] ** -damping
        sig += amp * np.sin(2 * np.pi * partial_freq * t + phases[h]) * np.exp(-t / partial_decay)

    sig = normalize(sig, 0.80)

    atk_dur = 0.007
    atk = noise(atk_dur, sr)
    atk = bandpass(atk, max(20, freq * 1.2), min(sr / 2 - 100, freq * 6), sr)
    atk = normalize(atk, 1.0) * env_exp_decay(atk_dur, 0.3, 0.0025, sr)

    body_exc = np.zeros(n)
    body_exc[:len(atk)] = atk
    delay = max(1, int(sr / freq))
    body = comb_filter(body_exc, delay, feedback=0.55, lp_freq=min(freq * 2.5, sr / 2 - 100), sr=sr)
    body *= env_exp_decay(dur, 0.5, decay_time * 0.5, sr)

    sig = sig + normalize(body[:n], 0.22)
    sig[:len(atk)] += atk * 0.28

    sig = asymmetric_saturate(sig, drive=1.1, asymmetry=0.05)
    sig = highpass(sig, 30, sr)
    sig = high_shelf(sig, 3000, env_profile['presence_db'], sr)
    sig = low_shelf(sig, 200, env_profile['warmth_db'], sr)

    sig = normalize(sig, 0.80)
    fade_in(sig, env_profile['attack_ms'], sr)
    fade_out(sig, 70, sr)

    return to_stereo(sig, pan) * np.sqrt(2)


def generate(out_dir, sr=SR, fmt='ogg'):
    freqs = FREQS
    print(f'  tuning: {SCALE} on {ROOT} -- {music.spell(ROOT, SCALE, TOP, len(freqs))}')
    print(f'  tone: {len(freqs)} pitches')
    segments = [gen_tone(freq, ENVELOPE, pan, sr)
                for freq, pan in zip(freqs, pan_positions(freqs))]

    entry = {'id': 'tone', 'label': 'tone', 'pitchCount': len(freqs), 'type': 'main'}
    entry.update(write_sprite(out_dir, 'tone', segments, sr, fmt))

    ir = generate_reverb_ir(2.0, dark=0.5, sr=sr)
    export_ogg(ir, os.path.join(out_dir, f'reverb_ir.{fmt}'), sr)

    config = dict(CONFIG)
    config['mainFreqs'] = list(freqs)
    write_manifest(out_dir, config, [entry], fmt=fmt)
