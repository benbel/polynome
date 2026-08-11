import numpy as np
from common import (
    SR, sine, noise, FORMANTS,
    env_adsr, env_exp_decay, lowpass, highpass, bandpass,
    normalize, fade_in, fade_out, f0_contour, glottal_flow, vocal_tract,
    scale_tract, fold_to_voice,
    to_stereo, export_ogg, write_sprite, write_manifest, generate_reverb_ir,
)

FREQS = [330, 294, 262, 233, 208, 185, 165, 147]
MEL_FREQS = [294, 262, 233, 208, 185, 165, 147, 131]

DURATIONS = {
    'vowels': 0.45, 'clicks': 0.20, 'hiss': 0.30,
    'nasal': 0.40, 'drone': 0.50, 'syllables': 0.70,
}

LOCUS = {
    'b': [(260, 90), (800, 130), (2200, 200), (3300, 250)],
    'd': [(260, 90), (1750, 150), (2600, 200), (3400, 250)],
    'g': [(260, 90), (2100, 160), (2500, 200), (3300, 250)],
    'l': [(320, 90), (1100, 140), (2600, 200), (3300, 250)],
    'm': [(280, 90), (1100, 180), (2200, 250), (3200, 300)],
    'n': [(280, 90), (1700, 200), (2600, 250), (3300, 300)],
}

BURST = {
    'b': (200, 900, 0.006, 0.10),
    'd': (2200, 5200, 0.004, 0.30),
    'g': (1200, 2600, 0.005, 0.22),
    'l': (300, 1600, 0.003, 0.06),
}

WORDS = [
    [('b', 'a'), ('b', 'a')],
    [('d', 'e'), ('g', 'u')],
    [('m', 'e'), ('n', 'o')],
    [('l', 'a'), ('b', 'a')],
    [('g', 'u'), ('m', 'e')],
    [('n', 'o'), ('l', 'a')],
    [('b', 'a'), ('d', 'e')],
    [('m', 'e'), ('l', 'a')],
]


def speech_f0(freq, dur, sr=SR, decline=-110):
    return f0_contour(fold_to_voice(freq, 90, 330), dur, sr,
                      vibrato_cents=0, jitter=0.012, drift_cents=decline)


def voiced_segment(freq, dur, start_tract, end_tract, sr=SR, decline=-60,
                   oq=0.62, aspiration=0.015):
    contour = speech_f0(freq, dur, sr, decline)
    source = glottal_flow(contour, sr, open_quotient=oq, shimmer=0.04,
                          aspiration=aspiration)
    return vocal_tract(source, start_tract, end_tract, sr, block=128)


def gen_vowels(freq, sr=SR):
    dur = DURATIONS['vowels']
    n = int(sr * dur)
    sig = voiced_segment(freq, dur, FORMANTS['schwa'], FORMANTS['a'], sr)
    sig *= env_adsr(dur, 0.012, 0.06, 0.8, 0.14, sr)[:n]
    sig = normalize(sig, 0.85)
    fade_in(sig, 3, sr)
    fade_out(sig, 30, sr)
    return to_stereo(sig, 0)


def gen_clicks(freq, sr=SR):
    dur = DURATIONS['clicks']
    n = int(sr * dur)
    sig = np.zeros(n)

    lo, hi, bdur, asp = BURST['d']
    burst = bandpass(noise(bdur, sr), lo, min(sr / 2 - 100, hi), sr)
    burst *= env_exp_decay(bdur, 0.15, 0.003, sr)[:len(burst)]
    sig[:len(burst)] = burst[:n]

    aspiration = bandpass(noise(0.025, sr), 1500, min(sr / 2 - 100, 5500), sr) * asp
    aspiration *= env_exp_decay(0.025, 0.2, 0.012, sr)[:len(aspiration)]
    end = min(len(burst) + len(aspiration), n)
    sig[len(burst):end] += aspiration[:end - len(burst)]

    vdur = 0.09
    voiced = voiced_segment(freq, vdur, LOCUS['d'], FORMANTS['e'], sr, decline=-40)
    voiced *= env_adsr(vdur, 0.006, 0.03, 0.6, 0.05, sr)[:len(voiced)] * 0.7
    start = len(burst) + int(sr * 0.012)
    end = min(start + len(voiced), n)
    sig[start:end] += voiced[:end - start]

    return to_stereo(normalize(sig, 0.85), 0)


def gen_hiss(freq, sr=SR):
    dur = DURATIONS['hiss']
    n = int(sr * dur)
    center = float(np.clip(5200 + (freq - 200) * 4, 3500, sr / 2 - 1200))
    source = highpass(noise(dur, sr), 2000, sr)
    sig = vocal_tract(source, [(center, 700), (center * 1.45, 900)], sr=sr)
    sig *= env_adsr(dur, 0.008, 0.04, 0.72, 0.06, sr)[:n]
    sig = normalize(sig, 0.85)
    fade_in(sig, 2, sr)
    fade_out(sig, 12, sr)
    return to_stereo(sig, 0)


def gen_nasal(freq, sr=SR):
    dur = DURATIONS['nasal']
    n = int(sr * dur)
    contour = speech_f0(freq, dur, sr, decline=-70)
    source = glottal_flow(contour, sr, open_quotient=0.5, shimmer=0.03)

    sig = vocal_tract(source, FORMANTS['n'], sr=sr)
    zero = vocal_tract(source, [(1250, 220)], sr=sr)
    peak = np.max(np.abs(zero))
    if peak > 0:
        zero = zero / peak * np.max(np.abs(sig)) * 0.45
    sig = sig[:n] - zero[:n]

    sig *= env_adsr(dur, 0.015, 0.06, 0.8, 0.1, sr)[:n]
    sig = normalize(sig, 0.85)
    fade_in(sig, 3, sr)
    fade_out(sig, 18, sr)
    return to_stereo(sig, 0)


def gen_drone(freq, sr=SR):
    dur = DURATIONS['drone']
    n = int(sr * dur)
    contour = speech_f0(freq, dur, sr, decline=-25)
    source = glottal_flow(contour, sr, open_quotient=0.65, shimmer=0.02)
    sig = vocal_tract(source, FORMANTS['o'], scale_tract(FORMANTS['o'], 1.04), sr)
    sig *= env_adsr(dur, 0.06, 0.1, 0.82, 0.16, sr)[:n]
    sig = normalize(sig, 0.85)
    fade_in(sig, 8, sr)
    fade_out(sig, 25, sr)
    return to_stereo(sig, 0)


def _syllable(freq, consonant, vowel, dur, decline, sr=SR):
    n = int(sr * dur)
    out = np.zeros(n)

    if consonant in ('m', 'n'):
        murmur = voiced_segment(freq, 0.06, FORMANTS[consonant], FORMANTS[consonant],
                                sr, decline=0, oq=0.5, aspiration=0) * 0.45
        end = min(len(murmur), n)
        out[:end] += murmur[:end]
        onset = int(sr * 0.05)
    else:
        lo, hi, bdur, asp = BURST[consonant]
        burst = bandpass(noise(bdur, sr), lo, min(sr / 2 - 100, hi), sr)
        burst *= env_exp_decay(bdur, 0.15, 0.003, sr)[:len(burst)]
        end = min(len(burst), n)
        out[:end] += burst[:end] * 0.8

        if asp > 0:
            aspiration = bandpass(noise(0.02, sr), 1500, min(sr / 2 - 100, 5000), sr) * asp
            aspiration *= env_exp_decay(0.02, 0.2, 0.01, sr)[:len(aspiration)]
            end = min(len(burst) + len(aspiration), n)
            out[len(burst):end] += aspiration[:end - len(burst)]
        onset = len(burst) + int(sr * 0.008)

    vdur = max(0.08, dur - onset / sr - 0.03)
    voiced = voiced_segment(freq, vdur, LOCUS[consonant], FORMANTS[vowel], sr,
                            decline=decline)
    voiced *= env_adsr(vdur, 0.012, 0.05, 0.75, 0.06, sr)[:len(voiced)]
    end = min(onset + len(voiced), n)
    out[onset:end] += voiced[:end - onset]
    return out


def gen_syllables(freq, sr=SR):
    dur = DURATIONS['syllables']
    n = int(sr * dur)
    word = WORDS[int(freq) % len(WORDS)]
    sig = np.zeros(n)
    syl_dur = dur / len(word)

    for si, (consonant, vowel) in enumerate(word):
        offset = int(sr * si * syl_dur)
        syl = _syllable(freq, consonant, vowel, syl_dur, -30 if si == 0 else -90, sr)
        end = min(offset + len(syl), n)
        sig[offset:end] += syl[:end - offset]

    sig *= env_adsr(dur, 0.005, 0.06, 0.85, 0.1, sr)[:n]
    sig = normalize(sig, 0.85)
    fade_in(sig, 2, sr)
    fade_out(sig, 30, sr)
    return to_stereo(sig, 0)


INSTRUMENTS = {
    'vowels': gen_vowels, 'clicks': gen_clicks, 'hiss': gen_hiss,
    'nasal': gen_nasal, 'drone': gen_drone, 'syllables': gen_syllables,
}
MEL_INSTS = {'syllables'}
LABELS = {inst_id: inst_id for inst_id in INSTRUMENTS}

FX_CONFIG = {
    'vowels': {'delay': 0.15, 'reverb': 0.25, 'gain': 0.24},
    'clicks': {'delay': 0.06, 'reverb': 0.08, 'gain': 0.30},
    'hiss': {'delay': 0.18, 'reverb': 0.25, 'gain': 0.20},
    'nasal': {'delay': 0.12, 'reverb': 0.20, 'gain': 0.26},
    'drone': {'delay': 0.10, 'reverb': 0.15, 'gain': 0.22},
    'syllables': {'delay': 0.15, 'reverb': 0.25, 'gain': 0.22},
}

CONFIG = {
    'id': 'speech',
    'label': 'speech',
    'mainGrids': [
        {'rows': 8, 'cols': 16, 'defaultInstrument': 'clicks'},
        {'rows': 8, 'cols': 16, 'defaultInstrument': 'vowels'},
        {'rows': 8, 'cols': 16, 'defaultInstrument': 'hiss'},
        {'rows': 8, 'cols': 16, 'defaultInstrument': 'drone'},
    ],
    'mainGridLayout': {'cols': 2},
    'melodyGrids': [
        {'rows': 8, 'cols': 32},
    ],
    'melodyDefaultInstrument': 'syllables',
    'effects': {
        'reverbWet': 0.18,
        'reverbDark': 0.5,
        'reverbLength': 2.0,
        'delayL': 0.35,
        'delayR': 0.23,
        'delayFeedback': 0.30,
        'delayDarkLP': 2500,
        'delayWet': 0.18,
        'compThreshold': -14,
        'compRatio': 5,
    },
    'numPatterns': 8,
    'defaultMelN': 2,
    'cellSize': 20,
    'fx': FX_CONFIG,
}


def generate(out_dir, sr=SR, fmt='ogg'):
    instruments = []

    for inst_id, gen_fn in INSTRUMENTS.items():
        is_mel = inst_id in MEL_INSTS
        freqs = MEL_FREQS if is_mel else FREQS
        print(f'  {inst_id}: {len(freqs)} pitches')

        segments = [gen_fn(freq, sr) for freq in freqs]

        entry = {
            'id': inst_id,
            'label': LABELS[inst_id],
            'pitchCount': len(freqs),
            'type': 'melody' if is_mel else 'main',
        }
        entry.update(write_sprite(out_dir, inst_id, segments, sr, fmt))
        instruments.append(entry)

    ir = generate_reverb_ir(2.0, dark=0.5, sr=sr)
    export_ogg(ir, f'{out_dir}/reverb_ir.{fmt}', sr)

    config = dict(CONFIG)
    config['mainFreqs'] = FREQS
    config['melodyFreqs'] = MEL_FREQS
    write_manifest(out_dir, config, instruments, fmt=fmt)
