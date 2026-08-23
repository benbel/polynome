import os
import numpy as np
import music
from common import (
    SR, sine, saw, pulse, noise, FORMANTS,
    env_adsr, env_exp_decay, lowpass, highpass, bandpass,
    moog_ladder, tanh_saturate, comb_filter,
    f0_contour, glottal_flow, vocal_tract, scale_tract, fold_to_voice,
    normalize, fade_in, fade_out, to_stereo, mix_stereo,
    export_ogg, write_sprite, write_manifest, generate_reverb_ir,
)

# One collection for the whole mode, so a melody note can never clash with
# whatever the rhythm grids happen to be sounding underneath it.
ROOT, SCALE = 'A', 'minor_pentatonic'

# Main grids, 16 rows top-down: A5 down to A2.  Exactly three octaves, so
# moving a cell five rows transposes it by an octave.  The previous table
# bottomed out at 25 Hz -- the lowest three rows were inaudible on any
# speaker without a subwoofer.
MAIN_TOP = 'A5'
FREQS = music.descending_hz(ROOT, SCALE, MAIN_TOP, 16)

# Melody grids, 8 rows: A5 down to E4, sitting in the upper half of the
# main range so the line floats above the bed rather than inside it.
MEL_TOP = 'A5'
MEL_FREQS = music.descending_hz(ROOT, SCALE, MEL_TOP, 8)

DURATIONS = {
    'moog_bass': 2.5, 'pluck': 1.8, 'string_machine': 3.0,
    'sync': 2.0, 'bell': 3.0, 'wurli': 2.5, 'clav': 1.5, 'celesta': 2.5,
    'aah': 2.8,
}


def gen_moog_bass(freq, sr=SR):
    f = freq
    dur = DURATIONS['moog_bass']
    n = int(sr * dur)

    sig = saw(f, dur, sr)

    cutoff = np.full(n, f * 0.6)
    env_n = int(sr * 0.4)
    cutoff[:env_n] = np.linspace(800, 80, env_n)
    cutoff[env_n:] = 80

    sig = moog_ladder(sig, cutoff, resonance=1.8, sr=sr)

    sub = sine(f * 0.5, dur, sr) * 0.3

    mix = sig * 0.7 + sub
    env = env_exp_decay(dur, 5, 0.8, sr)
    mix *= env

    result = normalize(mix, 0.85)
    fade_in(result, 5, sr)
    fade_out(result, 40, sr)
    return to_stereo(result, 0)


def gen_pluck(freq, sr=SR):
    f = freq
    dur = DURATIONS['pluck']
    n = int(sr * dur)

    exc_dur = 0.006
    exc = noise(exc_dur, sr) * 0.9
    exc = bandpass(exc, max(20, f * 0.8), min(sr / 2 - 100, f * 6), sr)
    exc_pad = np.zeros(n)
    exc_pad[:len(exc)] = exc

    delay1 = max(1, int(sr / f))
    c1 = comb_filter(exc_pad, delay1, feedback=0.92, lp_freq=min(f * 5, sr / 2 - 100), sr=sr)

    delay2 = max(1, int(sr / (f * 1.003)))
    c2 = comb_filter(exc_pad, delay2, feedback=0.90, lp_freq=min(f * 4, sr / 2 - 100), sr=sr)

    mix = c1 * 0.6 + c2 * 0.4

    atk = noise(0.003, sr)
    atk = highpass(atk, min(2000, sr / 2 - 100), sr) * 0.15
    mix[:len(atk)] += atk[:min(len(atk), n)]

    env = env_exp_decay(dur, 1, 0.5, sr)
    mix *= env

    result = normalize(mix, 0.85)
    fade_out(result, 25, sr)
    return to_stereo(result, 0)


def gen_string_machine(freq, sr=SR):
    f = freq
    dur = DURATIONS['string_machine']
    n = int(sr * dur)
    t = np.arange(n) / sr

    voices = []
    for i in range(8):
        detune = (i - 3.5) * 8
        inst_freq = f * (2 ** (detune / 1200))
        phase = np.cumsum(np.full(n, inst_freq) / sr)
        voice = 2 * (phase % 1) - 1
        pan = (i - 3.5) / 4
        voices.append(to_stereo(voice * 0.06, pan))

    mix = mix_stereo(*voices)[:n]

    cutoff = min(f * 6, sr / 2 - 100)
    for ch in range(2):
        mix[:, ch] = lowpass(mix[:, ch], cutoff, sr)

    env = env_adsr(dur, 0.2, 0.1, 0.8, 0.5, sr)
    mix *= env[:, np.newaxis]

    return normalize(mix, 0.85)


def gen_sync(freq, sr=SR):
    f = freq
    dur = DURATIONS['sync']
    n = int(sr * dur)
    t = np.arange(n) / sr

    ratio = 2.5 + 0.5 * np.sin(2 * np.pi * 0.8 * t)
    master_phase = np.cumsum(np.full(n, f) / sr)
    slave_freq = f * ratio
    slave_phase = np.cumsum(slave_freq / sr)

    master_cycle = master_phase % 1
    resets = np.diff(master_cycle) < -0.5
    resets = np.concatenate([[False], resets])
    cumulative_offset = np.zeros(n)
    for i in range(1, n):
        if resets[i]:
            cumulative_offset[i] = slave_phase[i]
        else:
            cumulative_offset[i] = cumulative_offset[i - 1]

    sig = np.sin(2 * np.pi * (slave_phase - cumulative_offset))

    sig = moog_ladder(sig, f * 4, resonance=1.0, sr=sr)
    env = env_exp_decay(dur, 3, 0.5, sr)
    sig *= env

    result = normalize(sig, 0.85)
    fade_in(result, 3, sr)
    fade_out(result, 30, sr)
    return to_stereo(result, 0)


def gen_bell(freq, sr=SR):
    f = freq
    dur = DURATIONS['bell']
    n = int(sr * dur)
    t = np.arange(n) / sr

    mod_index = np.linspace(5, 0.1, n)
    modulator = np.sin(2 * np.pi * f * 3.5 * t) * mod_index * f
    carrier = np.sin(2 * np.pi * f * t + modulator / f)

    p2 = np.sin(2 * np.pi * f * 5.3 * t) * 0.15
    p2_env = env_exp_decay(dur, 1, 0.3, sr)
    p3 = np.sin(2 * np.pi * f * 7.1 * t) * 0.08
    p3_env = env_exp_decay(dur, 1, 0.2, sr)

    sig = carrier * 0.7 + p2 * p2_env + p3 * p3_env

    env = env_exp_decay(dur, 2, 0.8, sr)
    sig *= env

    result = normalize(sig, 0.85)
    fade_in(result, 2, sr)
    fade_out(result, 80, sr)
    return to_stereo(result, 0)


def gen_wurli(freq, sr=SR):
    f = freq
    dur = DURATIONS['wurli']
    n = int(sr * dur)
    t = np.arange(n) / sr

    mod_index = np.linspace(3, 0.2, n) * f
    modulator = np.sin(2 * np.pi * f * t)
    sig = np.sin(2 * np.pi * f * t + modulator * mod_index / f)

    trem = 1 + 0.1 * np.sin(2 * np.pi * 4.5 * t)
    sig *= trem

    env = env_exp_decay(dur, 5, 0.5, sr)
    sig *= env

    result = normalize(sig, 0.85)
    fade_in(result, 5, sr)
    fade_out(result, 40, sr)
    return to_stereo(result, 0)


def gen_clav(freq, sr=SR):
    f = freq
    dur = DURATIONS['clav']
    n = int(sr * dur)

    exc = noise(0.003, sr)
    exc = highpass(exc, 1000, sr) * 0.8
    exc_pad = np.zeros(n)
    exc_pad[:len(exc)] = exc

    delay = max(1, int(sr / f))
    sig = comb_filter(exc_pad, delay, feedback=0.9, lp_freq=min(f * 5, sr / 2 - 100), sr=sr)

    env = env_exp_decay(dur, 0.5, 0.2, sr)
    sig *= env

    result = normalize(sig, 0.85)
    fade_out(result, 20, sr)
    return to_stereo(result, 0)


def gen_celesta(freq, sr=SR):
    f = freq
    dur = DURATIONS['celesta']
    n = int(sr * dur)
    t = np.arange(n) / sr

    partials = [
        (1.0,   0.50, 0.5),
        (2.76,  0.35, 0.35),
        (3.01,  0.25, 0.3),
        (5.40,  0.18, 0.2),
        (8.93,  0.10, 0.12),
        (13.2,  0.05, 0.08),
    ]

    sig = np.zeros(n)
    for ratio, amp, decay_time in partials:
        pf = f * ratio
        if pf >= sr / 2:
            continue
        partial = np.sin(2 * np.pi * pf * t) * amp
        p_env = env_exp_decay(dur, 1, decay_time, sr)
        sig += partial * p_env

    click = noise(0.002, sr)
    click = highpass(click, min(3000, sr / 2 - 100), sr) * 0.12
    sig[:len(click)] += click[:min(len(click), n)]

    result = normalize(sig, 0.85)
    fade_in(result, 1, sr)
    fade_out(result, 50, sr)
    return to_stereo(result, 0)


def gen_aah(freq, sr=SR):
    """Sung 'ah' -- the one voice kept from the retired voices mode.

    The glottal source is folded into a singable register before the
    formant filters run, so it tracks the grid row by octave rather than
    trying to sing A2.
    """
    dur = DURATIONS['aah']
    n = int(sr * dur)

    contour = f0_contour(fold_to_voice(freq), dur, sr,
                         vibrato_rate=5.0, vibrato_cents=14, jitter=0.004)
    source = glottal_flow(contour, sr, open_quotient=0.6, shimmer=0.025,
                          aspiration=0.018)
    sig = vocal_tract(source,
                      scale_tract(FORMANTS['schwa'], 1.0),
                      scale_tract(FORMANTS['a'], 1.0), sr)

    sig *= env_adsr(dur, 0.14, 0.25, 0.74, 0.6, sr)[:n]
    return normalize(to_stereo(sig[:n], 0), 0.85)


INSTRUMENTS = {
    'moog_bass': gen_moog_bass, 'pluck': gen_pluck,
    'string_machine': gen_string_machine, 'sync': gen_sync, 'bell': gen_bell,
    'wurli': gen_wurli, 'clav': gen_clav, 'celesta': gen_celesta,
    'aah': gen_aah,
}
MEL_INSTS = {'wurli', 'clav', 'celesta', 'aah'}

FX_CONFIG = {
    'moog_bass': {'delay': 0.08, 'reverb': 0.12, 'gain': 0.30},
    'pluck': {'delay': 0.10, 'reverb': 0.12, 'gain': 0.26},
    'string_machine': {'delay': 0.20, 'reverb': 0.35, 'gain': 0.18},
    'sync': {'delay': 0.12, 'reverb': 0.18, 'gain': 0.22},
    'bell': {'delay': 0.30, 'reverb': 0.45, 'gain': 0.16},
    'wurli': {'delay': 0.18, 'reverb': 0.25, 'gain': 0.20},
    'clav': {'delay': 0.10, 'reverb': 0.15, 'gain': 0.24},
    'celesta': {'delay': 0.25, 'reverb': 0.40, 'gain': 0.18},
    'aah': {'delay': 0.12, 'reverb': 0.28, 'gain': 0.20},
}


LABELS = {
    'moog_bass': 'moog bass', 'pluck': 'pluck', 'string_machine': 'strings',
    'sync': 'sync', 'bell': 'bell', 'wurli': 'wurli', 'clav': 'clav',
    'celesta': 'celesta', 'aah': 'voice',
}

CONFIG = {
    'id': 'ensemble',
    'label': 'ensemble',
    'mainGrids': [
        {'rows': 16, 'cols': 32, 'defaultInstrument': 'moog_bass'},
        {'rows': 16, 'cols': 32, 'defaultInstrument': 'pluck'},
        {'rows': 16, 'cols': 32, 'defaultInstrument': 'string_machine'},
        {'rows': 16, 'cols': 32, 'defaultInstrument': 'bell'},
    ],
    'mainGridLayout': {'cols': 2},
    'melodyGrids': [
        {'rows': 8, 'cols': 32},
        {'rows': 8, 'cols': 32},
    ],
    'melodyDefaultInstrument': 'wurli',
    'effects': {
        'reverbWet': 0.08,
        'reverbDark': 0.3,
        'reverbLength': 1.8,
        'delayL': 0.38,
        'delayR': 0.25,
        'delayFeedback': 0.18,
        'delayDarkLP': 5000,
        'delayWet': 0.08,
        'compThreshold': -20,
        'compRatio': 2.5,
    },
    'numPatterns': 8,
    'defaultMelN': 2,
    'cellSize': 16,
    'fx': FX_CONFIG,
}


def generate(out_dir, sr=SR, fmt='ogg'):
    instruments = []

    print(f'  tuning: {SCALE} on {ROOT}')
    print(f'    main   {music.spell(ROOT, SCALE, MAIN_TOP, len(FREQS))}')
    print(f'    melody {music.spell(ROOT, SCALE, MEL_TOP, len(MEL_FREQS))}')

    for inst_id, gen_fn in INSTRUMENTS.items():
        is_mel = inst_id in MEL_INSTS
        freqs = MEL_FREQS if is_mel else FREQS

        print(f'  {inst_id}: {len(freqs)} pitches')
        segments = [normalize(gen_fn(freq, sr), 0.85) for freq in freqs]

        entry = {
            'id': inst_id,
            'label': LABELS[inst_id],
            'pitchCount': len(freqs),
            'type': 'melody' if is_mel else 'main',
        }
        entry.update(write_sprite(out_dir, inst_id, segments, sr, fmt))
        instruments.append(entry)

    ir = generate_reverb_ir(2.5, dark=0.4, sr=sr)
    export_ogg(ir, os.path.join(out_dir, f'reverb_ir.{fmt}'), sr)

    config = dict(CONFIG)
    config['mainFreqs'] = FREQS
    config['melodyFreqs'] = MEL_FREQS
    write_manifest(out_dir, config, instruments, fmt=fmt)
