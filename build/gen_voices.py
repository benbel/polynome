import numpy as np
from common import (
    SR, sine, noise, FORMANTS,
    env_adsr, env_exp_decay, lowpass, highpass, bandpass,
    normalize, fade_in, fade_out, f0_contour, glottal_flow, vocal_tract,
    scale_tract, fold_to_voice, smooth_noise,
    to_stereo, export_ogg, write_sprite, write_manifest, generate_reverb_ir,
)

FREQS = [659, 587, 523, 466, 415, 370, 330, 294,
         262, 233, 208, 185, 165, 147, 131, 110]
MEL_FREQS = [523, 466, 415, 370, 330, 294, 262, 220]

DURATIONS = {
    'throat': 3.0, 'overtone': 3.0, 'breath': 2.5, 'belt': 2.5, 'click': 0.8,
    'aah': 2.8, 'ooh': 2.8, 'mmm': 2.8,
}

TRACT = {
    'throat': 1.18, 'overtone': 1.10, 'breath': 1.0, 'belt': 0.94,
    'aah': 1.0, 'ooh': 1.02, 'mmm': 1.05,
}


def sung(freq, dur, vowel_from, vowel_to, tract, sr=SR,
         oq=0.6, vibrato=18, rate=5.0, aspiration=0.02, drift=0):
    f0 = fold_to_voice(freq)
    contour = f0_contour(f0, dur, sr, vibrato_rate=rate, vibrato_cents=vibrato,
                         jitter=0.004, drift_cents=drift)
    source = glottal_flow(contour, sr, open_quotient=oq, shimmer=0.025,
                          aspiration=aspiration)
    return vocal_tract(source,
                       scale_tract(FORMANTS[vowel_from], tract),
                       scale_tract(FORMANTS[vowel_to], tract), sr)


def gen_throat(freq, sr=SR):
    dur = DURATIONS['throat']
    n = int(sr * dur)
    sig = sung(freq, dur, 'u', 'o', TRACT['throat'], sr,
               oq=0.48, vibrato=12, rate=4.4, aspiration=0.015)
    sub = sine(fold_to_voice(freq) / 2, dur, sr) * 0.12
    sig = sig[:n] + sub[:n]
    sig *= env_adsr(dur, 0.18, 0.3, 0.72, 0.6, sr)[:n]
    return normalize(to_stereo(sig, 0), 0.85)


def gen_overtone(freq, sr=SR):
    dur = DURATIONS['overtone']
    n = int(sr * dur)
    f0 = fold_to_voice(freq)
    contour = f0_contour(f0, dur, sr, vibrato_rate=3.8, vibrato_cents=8, jitter=0.003)
    source = glottal_flow(contour, sr, open_quotient=0.55, shimmer=0.02, aspiration=0.01)

    base = vocal_tract(source, scale_tract(FORMANTS['u'], TRACT['overtone']), sr=sr)

    harmonic = 8 if f0 < 160 else 6 if f0 < 260 else 4
    overtone = vocal_tract(source, [(min(f0 * harmonic, sr / 2 - 400), 45)], sr=sr)
    peak = np.max(np.abs(overtone))
    if peak > 0:
        overtone = overtone / peak * np.max(np.abs(base)) * 0.9

    sig = base[:n] + overtone[:n]
    sig *= env_adsr(dur, 0.25, 0.3, 0.7, 0.7, sr)[:n]
    return normalize(to_stereo(sig, 0), 0.85)


def gen_breath(freq, sr=SR):
    dur = DURATIONS['breath']
    n = int(sr * dur)
    contour = f0_contour(fold_to_voice(freq), dur, sr, vibrato_cents=6, jitter=0.008)
    voiced = glottal_flow(contour, sr, open_quotient=0.8, shimmer=0.05) * 0.12
    air = highpass(np.random.uniform(-1, 1, n), 400, sr) * 0.9

    sig = vocal_tract(voiced[:n] + air,
                      scale_tract(FORMANTS['e'], TRACT['breath']),
                      scale_tract(FORMANTS['a'], TRACT['breath']), sr)
    sig = lowpass(sig, 6500, sr)
    sig *= env_adsr(dur, 0.35, 0.3, 0.6, 0.7, sr)[:n] * 0.6
    return normalize(to_stereo(sig, 0), 0.85)


def gen_belt(freq, sr=SR):
    dur = DURATIONS['belt']
    n = int(sr * dur)
    sig = sung(freq, dur, 'e', 'a', TRACT['belt'], sr,
               oq=0.72, vibrato=26, rate=5.6, aspiration=0.012)

    singers = vocal_tract(sig, [(2900, 160)], sr=sr)
    peak = np.max(np.abs(singers))
    if peak > 0:
        singers = singers / peak * np.max(np.abs(sig)) * 0.35
    sig = sig[:n] + singers[:n]

    sig *= env_adsr(dur, 0.05, 0.15, 0.88, 0.4, sr)[:n]
    return normalize(to_stereo(sig, 0), 0.85)


def gen_click(freq, sr=SR):
    dur = DURATIONS['click']
    n = int(sr * dur)
    f0 = fold_to_voice(freq)
    sig = np.zeros(n)

    if f0 > 300:
        cavity = [(1800, 200), (3200, 300)]
    elif f0 > 160:
        cavity = [(900, 150), (2100, 250)]
    else:
        cavity = [(420, 110), (1300, 200)]

    burst_len = int(sr * 0.004)
    burst = np.random.uniform(-1, 1, burst_len) * np.linspace(1, 0, burst_len) ** 2
    click = vocal_tract(np.concatenate([burst, np.zeros(int(sr * 0.12))]), cavity, sr=sr)
    click *= env_exp_decay(len(click) / sr, 0.2, 0.03, sr)[:len(click)]
    sig[:min(len(click), n)] = click[:n]

    tail = glottal_flow(f0_contour(f0, 0.07, sr, jitter=0.01), sr, open_quotient=0.5)
    tail = vocal_tract(tail, cavity, sr=sr) * 0.25
    tail *= env_exp_decay(0.07, 0.5, 0.02, sr)[:len(tail)]
    end = min(burst_len + len(tail), n)
    sig[burst_len:end] += tail[:end - burst_len]

    return normalize(to_stereo(sig, 0), 0.85)


def _vowel(freq, key, vowel, sr, oq=0.6, vibrato=14):
    dur = DURATIONS[key]
    n = int(sr * dur)
    sig = sung(freq, dur, 'schwa', vowel, TRACT[key], sr,
               oq=oq, vibrato=vibrato, rate=5.0, aspiration=0.018)
    sig *= env_adsr(dur, 0.14, 0.25, 0.74, 0.6, sr)[:n]
    return normalize(to_stereo(sig, 0), 0.85)


def gen_aah(freq, sr=SR):
    return _vowel(freq, 'aah', 'a', sr)


def gen_ooh(freq, sr=SR):
    return _vowel(freq, 'ooh', 'u', sr, oq=0.55, vibrato=11)


def gen_mmm(freq, sr=SR):
    dur = DURATIONS['mmm']
    n = int(sr * dur)
    contour = f0_contour(fold_to_voice(freq), dur, sr,
                         vibrato_rate=4.6, vibrato_cents=9, jitter=0.004)
    source = glottal_flow(contour, sr, open_quotient=0.45, shimmer=0.02)

    sig = vocal_tract(source,
                      scale_tract(FORMANTS['n'], TRACT['mmm']),
                      scale_tract(FORMANTS['m'], TRACT['mmm']), sr)
    sig = lowpass(sig, 2200, sr)
    sig *= env_adsr(dur, 0.16, 0.25, 0.78, 0.6, sr)[:n]
    return normalize(to_stereo(sig, 0), 0.85)


INSTRUMENTS = {
    'throat': gen_throat, 'overtone': gen_overtone, 'breath': gen_breath,
    'belt': gen_belt, 'click': gen_click,
    'aah': gen_aah, 'ooh': gen_ooh, 'mmm': gen_mmm,
}
MEL_INSTS = {'aah', 'ooh', 'mmm'}
LABELS = {inst_id: inst_id for inst_id in INSTRUMENTS}

FX_CONFIG = {
    'throat': {'delay': 0.08, 'reverb': 0.10, 'gain': 0.26},
    'overtone': {'delay': 0.10, 'reverb': 0.15, 'gain': 0.22},
    'breath': {'delay': 0.12, 'reverb': 0.18, 'gain': 0.18},
    'belt': {'delay': 0.06, 'reverb': 0.08, 'gain': 0.28},
    'click': {'delay': 0.06, 'reverb': 0.08, 'gain': 0.30},
    'aah': {'delay': 0.10, 'reverb': 0.20, 'gain': 0.22},
    'ooh': {'delay': 0.10, 'reverb': 0.20, 'gain': 0.20},
    'mmm': {'delay': 0.08, 'reverb': 0.15, 'gain': 0.22},
}

CONFIG = {
    'id': 'voices',
    'label': 'voices',
    'mainGrids': [
        {'rows': 16, 'cols': 32, 'defaultInstrument': 'throat'},
        {'rows': 16, 'cols': 32, 'defaultInstrument': 'overtone'},
        {'rows': 16, 'cols': 32, 'defaultInstrument': 'breath'},
        {'rows': 16, 'cols': 32, 'defaultInstrument': 'belt'},
    ],
    'mainGridLayout': {'cols': 2},
    'melodyGrids': [
        {'rows': 8, 'cols': 32},
        {'rows': 8, 'cols': 32},
    ],
    'melodyDefaultInstrument': 'aah',
    'effects': {
        'reverbWet': 0.15,
        'reverbDark': 0.45,
        'reverbLength': 2.5,
        'delayL': 0.40,
        'delayR': 0.28,
        'delayFeedback': 0.20,
        'delayDarkLP': 3000,
        'delayWet': 0.12,
        'compThreshold': -18,
        'compRatio': 3,
    },
    'numPatterns': 8,
    'defaultMelN': 2,
    'cellSize': 16,
    'fx': FX_CONFIG,
}


def generate(out_dir, sr=SR, fmt='ogg'):
    instruments = []

    for inst_id, gen_fn in INSTRUMENTS.items():
        is_mel = inst_id in MEL_INSTS
        freqs = MEL_FREQS if is_mel else FREQS
        print(f'  {inst_id}: {len(freqs)} pitches')

        segments = []
        for freq in freqs:
            sig = gen_fn(freq, sr)
            fade_in(sig, 12, sr)
            fade_out(sig, 90, sr)
            segments.append(sig)

        entry = {
            'id': inst_id,
            'label': LABELS[inst_id],
            'pitchCount': len(freqs),
            'type': 'melody' if is_mel else 'main',
        }
        entry.update(write_sprite(out_dir, inst_id, segments, sr, fmt))
        instruments.append(entry)

    ir = generate_reverb_ir(2.5, dark=0.45, sr=sr)
    export_ogg(ir, f'{out_dir}/reverb_ir.{fmt}', sr)

    config = dict(CONFIG)
    config['mainFreqs'] = FREQS
    config['melodyFreqs'] = MEL_FREQS
    write_manifest(out_dir, config, instruments, fmt=fmt)
