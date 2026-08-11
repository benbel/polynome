import os
import numpy as np
from common import (
    SR, sine, saw, pulse, noise, stereo_noise,
    env_exp_decay, env_adsr, lowpass, highpass, bandpass, filter_sweep,
    wavefold, tanh_saturate, asymmetric_saturate, bitcrush, comb_filter,
    normalize, fade_in, fade_out, to_stereo, mix_stereo,
    export_ogg, write_sprite, write_manifest, generate_reverb_ir,
)

FREQS = [523, 440, 370, 311, 262, 220, 175, 147, 123, 104, 82, 65, 55, 44, 33, 25]
MEL_FREQS = [659, 523, 440, 349, 262, 220, 175, 131]

DURATIONS = {
    'sub': 2.0, 'fm': 1.6, 'glass': 2.6, 'tape': 2.4, 'dust': 1.2,
    'pad': 3.5, 'organ': 2.8, 'piano': 2.4,
}

CRUSH = {
    'sub': (8, 5, 2.5), 'fm': (10, 3, 2.0), 'glass': (14, 1, 1.1),
    'tape': (9, 5, 3.2), 'dust': (9, 3, 2.0), 'pad': (13, 1, 1.2),
    'organ': (12, 2, 1.4), 'piano': (15, 1, 1.0),
}

FADE = {
    'sub': (40, 80), 'fm': (20, 40), 'glass': (20, 50), 'tape': (35, 70),
    'dust': (8, 20), 'pad': (60, 120), 'organ': (15, 40), 'piano': (5, 30),
}


def gen_sub(freq, sr=SR):
    f = freq * 0.25
    dur = DURATIONS['sub']
    n = int(sr * dur)

    voices = []
    detunes = [-28, -18, -8, -3, 3, 8, 18, 28]
    pans = [-0.9, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 0.9]
    drift_rates = [0.13, 0.09, 0.17, 0.11, 0.14, 0.08, 0.16, 0.1]

    for i in range(8):
        t = np.arange(n) / sr
        drift = np.sin(2 * np.pi * drift_rates[i] * t) * (drift_rates[i] * 20)
        inst_freq = f * (2 ** ((detunes[i] + drift) / 1200))
        phase = np.cumsum(inst_freq / sr)
        if i < 4:
            voice = 2 * (phase % 1) - 1
        else:
            voice = np.where(phase % 1 < 0.5, 1.0, -1.0)
        gain = 0.14 if i < 4 else 0.10
        voices.append(to_stereo(voice * gain, pans[i]))

    mix = mix_stereo(*voices) * 0.4

    for ch in range(2):
        x = mix[:, ch] * 3
        folded = (4 / np.pi) * np.arcsin(np.sin(np.pi * x / 2))
        mix[:, ch] = np.where(x > 0, folded * 0.85, folded)

    mix = np.tanh(mix * 6) / np.tanh(6)
    for ch in range(2):
        mix[:, ch] = filter_sweep(mix[:, ch], f * 16, f * 1.3, sr, order=4)

    sub_sig = to_stereo(sine(f, dur, sr) * 0.5, 0)
    hum = to_stereo(sine(50, dur, sr) * 0.06, 0)

    noise_sig = noise(0.5, sr)
    noise_sig = wavefold(noise_sig, 2)
    noise_sig = lowpass(noise_sig, f * 8, sr)
    noise_env = env_exp_decay(0.5, 1, 0.15, sr)
    noise_sig *= noise_env * 0.35
    noise_stereo = to_stereo(noise_sig, 0)

    result = mix_stereo(mix[:n], sub_sig[:n], hum[:n], noise_stereo)
    env = env_exp_decay(dur, 3, 0.8, sr)
    result *= env[:, np.newaxis]
    return result


def gen_fm(freq, sr=SR):
    f = freq
    dur = DURATIONS['fm']
    n = int(sr * dur)
    t = np.arange(n) / sr

    drift1 = np.sin(2 * np.pi * 0.11 * t) * 1.5
    drift2 = np.sin(2 * np.pi * 0.08 * t) * 2.0
    car1_freq = f * (2 ** (drift1 / 1200))
    car2_freq = f * 1.005 * (2 ** (drift2 / 1200))

    m1_idx = np.linspace(4, 0.2, n) * f
    m2_idx = np.linspace(2.5, 0.15, n) * f
    m3_idx = np.linspace(1.2, 0.5, n) * f

    m1 = np.sin(2 * np.pi * f * 1.4142 * t) * m1_idx
    m2 = np.sin(2 * np.pi * f * 1.618 * t) * m2_idx
    m3 = np.sin(2 * np.pi * f * 0.5 * t) * m3_idx

    car1 = np.sin(2 * np.pi * np.cumsum(car1_freq + m1 + m2 + m3) / sr) * 0.55
    car2 = np.sin(2 * np.pi * np.cumsum(car2_freq + m1 + m3) / sr) * 0.4

    mix = to_stereo(car1, -0.35)[:n] + to_stereo(car2, 0.35)[:n]
    for ch in range(2):
        x = mix[:, ch] * 1.8
        folded = (4 / np.pi) * np.arcsin(np.sin(np.pi * x / 2))
        mix[:, ch] = np.where(x > 0, folded * 0.8, folded * 1.1)

    for ch in range(2):
        mix[:, ch] = filter_sweep(mix[:, ch], f * 6, f * 1.2, sr)
    hum = to_stereo(sine(50, dur, sr) * 0.03, 0)

    result = mix_stereo(mix, hum[:n])
    env = env_exp_decay(dur, 2, 0.6, sr)
    result *= env[:, np.newaxis]
    return result


def gen_glass(freq, sr=SR):
    f = freq
    dur = DURATIONS['glass']
    n = int(sr * dur)

    exc = noise(0.02, sr)
    result = np.zeros((n, 2))

    freqs = [f, f*1.5, f*2.76, f*3.51, f*4.23, f*5.87, f*7.1]
    amps = [0.45, 0.25, 0.25, 0.18, 0.12, 0.08, 0.05]
    pan_vals = [-0.5, -0.2, 0.1, 0.35, -0.6, 0.55, -0.8]
    decays = [2.2, 1.8, 1.5, 1.2, 0.8, 0.5, 0.3]

    for i in range(len(freqs)):
        bp = bandpass(exc, max(20, freqs[i] - freqs[i]/10), min(sr/2 - 100, freqs[i] + freqs[i]/10), sr)
        padded = np.zeros(n)
        padded[:len(bp)] = bp
        env = env_exp_decay(dur, 1, decays[i] / 3, sr)
        padded *= env * amps[i]
        result += to_stereo(padded, pan_vals[i])[:n]

    exc_long = np.zeros(n)
    exc_long[:len(exc)] = exc * 0.8
    delay1 = max(1, int(sr / f))
    c1 = comb_filter(exc_long, delay1, feedback=0.72, lp_freq=f * 6, sr=sr)
    c1_env = env_exp_decay(dur, 1, 0.7, sr)
    result += to_stereo(c1 * c1_env * 0.35, 0)[:n]

    body = sine(f, dur, sr) * wavefold(sine(f, dur, sr), 1.5) * 0.25
    body_env = env_exp_decay(dur, 1, 0.6, sr)
    result += to_stereo(body * body_env, 0)[:n]

    env = env_exp_decay(dur, 1, 0.8, sr)
    result *= env[:, np.newaxis]
    return result


def gen_tape(freq, sr=SR):
    f = freq
    dur = DURATIONS['tape']
    n = int(sr * dur)
    t = np.arange(n) / sr

    voices = []
    detunes = [-15, -8, -3, 3, 8, 15]
    pans = [-0.7, -0.35, -0.1, 0.1, 0.35, 0.7]
    wow_rates = [0.4, 0.35, 0.55, 0.45, 0.6, 0.38]
    wow_depths = [8, 12, 6, 14, 9, 11]

    for i in range(6):
        wow = np.sin(2 * np.pi * wow_rates[i] * t) * wow_depths[i]
        flutter = np.sin(2 * np.pi * (4.5 + i * 0.3) * t) * 2.5
        inst_freq = f * (2 ** ((detunes[i] + wow + flutter) / 1200))
        voice = np.sin(2 * np.pi * np.cumsum(inst_freq / sr))
        voices.append(to_stereo(voice * 0.18, pans[i]))

    mix = mix_stereo(*voices)[:n]

    for ch in range(2):
        x = mix[:, ch]
        mix[:, ch] = np.where(x > 0, np.tanh(x * 3.0) / 3.0 * 2.5, np.tanh(x * 2.0) / 2.0)

    for ch in range(2):
        mix[:, ch] = lowpass(mix[:, ch], min(f * 3.5, sr / 2 - 100), sr)

    hiss = stereo_noise(dur, sr) * 0.10
    for ch in range(2):
        hiss[:, ch] = lowpass(hiss[:, ch], min(f * 4, sr / 2 - 100), sr)
        hiss[:, ch] = highpass(hiss[:, ch], max(20, f * 0.5), sr)

    dropout_env = np.ones(n)
    for _ in range(3):
        pos = np.random.randint(int(n * 0.1), int(n * 0.8))
        width = np.random.randint(int(sr * 0.01), int(sr * 0.04))
        end = min(pos + width, n)
        dropout_env[pos:end] *= 0.3 + np.random.random() * 0.4

    result = mix_stereo(mix, hiss[:n])
    result *= dropout_env[:, np.newaxis]
    env = env_adsr(dur, 0.04, 0.1, 0.7, 0.3, sr)
    result *= env[:, np.newaxis]
    return result


def gen_dust(freq, sr=SR):
    f = freq * 0.5
    dur = DURATIONS['dust']
    n = int(sr * dur)

    result = np.zeros((n, 2))

    delay = max(1, int(sr / f))
    exc = noise(0.01, sr)
    exc_pad = np.zeros(n)
    exc_pad[:len(exc)] = exc
    ks = comb_filter(exc_pad, delay, feedback=0.85, lp_freq=f * 3, sr=sr)
    ks_env = env_exp_decay(dur, 1, 0.3, sr)
    result += to_stereo(ks * ks_env * 0.35, 0)[:n]

    ratios = [1, 1.34, 1.87]
    comb_pans = [0, -0.5, 0.5]
    for i, ratio in enumerate(ratios):
        d = max(1, int(sr / (f * ratio)))
        grain = noise(0.005, sr)
        grain_pad = np.zeros(n)
        grain_pad[:len(grain)] = grain * 0.4
        c = comb_filter(grain_pad, d, feedback=0.6, sr=sr)
        c_env = env_exp_decay(dur, 1, 0.15 + i * 0.05, sr)
        result += to_stereo(c * c_env * 0.2, comb_pans[i])[:n]

    for count, offset_base in [(12, 0), (6, 0.035), (3, 0.07)]:
        for _ in range(count):
            g_dur = 0.002 + np.random.random() * 0.01
            g = noise(g_dur, sr)
            g = bandpass(g, max(20, f * 0.5), min(sr/2 - 100, f * 6), sr)
            g = wavefold(g, 1.2 + np.random.random() * 1.5)
            g *= 0.1 + np.random.random() * 0.2
            start = int((offset_base + np.random.random() * 0.025) * sr)
            pan = (np.random.random() - 0.5) * 1.6
            gs = to_stereo(g, pan)
            end = min(start + len(gs), n)
            if start < n:
                result[start:end] += gs[:end - start]

    tail = noise(0.6, sr)
    tail = lowpass(tail, f * 2.5, sr)
    tail = wavefold(tail, 1.5)
    tail_env = env_exp_decay(0.6, 1, 0.2, sr)
    tail *= tail_env * 0.2
    tail_s = to_stereo(tail, 0)
    result[:len(tail_s)] += tail_s

    return result


def gen_pad(freq, sr=SR):
    f = freq
    dur = DURATIONS['pad']
    n = int(sr * dur)
    t = np.arange(n) / sr

    detunes = [-25, -18, -10, -4, 4, 10, 18, 25]
    pans = [-0.9, -0.65, -0.35, -0.1, 0.1, 0.35, 0.65, 0.9]
    voices = []
    for i in range(8):
        inst_freq = f * (2 ** (detunes[i] / 1200))
        phase = np.cumsum(np.full(n, inst_freq) / sr)
        voice = 2 * (phase % 1) - 1
        voices.append(to_stereo(voice * 0.08, pans[i]))

    mix = mix_stereo(*voices)[:n]

    for ch in range(2):
        mix[:, ch] = wavefold(mix[:, ch], 1.5)
    lfo = f * 3 + np.sin(2 * np.pi * 0.2 * t) * f * 2
    for ch in range(2):
        mix[:, ch] = filter_sweep(mix[:, ch], f, f * 5, sr)

    sub = to_stereo(sine(f, dur, sr) * 0.18, 0)
    result = mix_stereo(mix, sub[:n])

    env = env_adsr(dur, 0.8, 0.2, 0.8, 1.2, sr)
    result *= env[:, np.newaxis]
    return result


def gen_organ(freq, sr=SR):
    f = freq
    dur = DURATIONS['organ']
    n = int(sr * dur)
    t = np.arange(n) / sr

    drawbars = [1, 2, 3, 4, 6, 8]
    amps = [0.8, 0.6, 0.3, 0.2, 0.15, 0.1]
    mix = np.zeros(n)
    for i, db in enumerate(drawbars):
        harm_freq = f * db
        if harm_freq < sr / 2:
            mix += sine(harm_freq, dur, sr) * amps[i] * 0.15

    trem = 1 + np.sin(2 * np.pi * 5.8 * t) * 0.2
    mix *= trem

    mix = np.where(mix > 0, np.tanh(mix * 1.8) / 1.8 * 1.5, np.tanh(mix * 1.3) / 1.3)

    click = noise(0.005, sr)
    click = highpass(click, 2000, sr) * 0.08
    result_mono = np.zeros(n)
    result_mono[:len(click)] += click
    result_mono += mix

    result = to_stereo(result_mono, 0)
    env = env_adsr(dur, 0.06, 0.1, 0.8, 0.8, sr)
    result *= env[:, np.newaxis]
    return result


def gen_piano(freq, sr=SR):
    f = freq
    dur = DURATIONS['piano']
    n = int(sr * dur)

    exc = noise(0.008, sr)
    exc = bandpass(exc, max(20, f), min(sr/2 - 100, f * 4), sr) * 0.7
    exc_pad = np.zeros(n)
    exc_pad[:len(exc)] = exc

    delay1 = max(1, int(sr / f))
    c1 = comb_filter(exc_pad, delay1, feedback=0.92, lp_freq=f * 4, sr=sr)
    c1_env = env_exp_decay(dur, 1, 0.8, sr)

    delay2 = max(1, int(sr / (f * 2.01)))
    c2 = comb_filter(exc_pad, delay2, feedback=0.88, lp_freq=f * 3, sr=sr)
    c2_env = env_exp_decay(dur, 1, 0.4, sr)

    result = to_stereo(c1 * c1_env * 0.4, 0)[:n] + to_stereo(c2 * c2_env * 0.2, 0.3)[:n]

    thump = sine(f * 1.2, 0.04, sr)
    thump_env = env_exp_decay(0.04, 0.5, 0.01, sr)
    thump *= thump_env * 0.15
    thump_s = to_stereo(thump, 0)
    result[:len(thump_s)] += thump_s

    for ch in range(2):
        result[:, ch] = filter_sweep(result[:, ch], f * 6, f * 2, sr)

    env = env_exp_decay(dur, 1, 0.7, sr)
    result *= env[:, np.newaxis]
    return result


INSTRUMENTS = {
    'sub': gen_sub, 'fm': gen_fm, 'glass': gen_glass, 'tape': gen_tape,
    'dust': gen_dust, 'pad': gen_pad, 'organ': gen_organ, 'piano': gen_piano,
}

MEL_INSTS = {'pad', 'organ', 'piano'}

FX_CONFIG = {
    'sub': {'delay': 0.06, 'reverb': 0.10, 'gain': 0.34},
    'fm': {'delay': 0.18, 'reverb': 0.25, 'gain': 0.22},
    'glass': {'delay': 0.28, 'reverb': 0.40, 'gain': 0.20},
    'tape': {'delay': 0.22, 'reverb': 0.35, 'gain': 0.20},
    'dust': {'delay': 0.14, 'reverb': 0.22, 'gain': 0.28},
    'pad': {'delay': 0.25, 'reverb': 0.50, 'gain': 0.16},
    'organ': {'delay': 0.18, 'reverb': 0.30, 'gain': 0.18},
    'piano': {'delay': 0.22, 'reverb': 0.38, 'gain': 0.20},
}


LABELS = {inst_id: inst_id for inst_id in INSTRUMENTS}

CONFIG = {
    'id': 'texture',
    'label': 'texture',
    'mainGrids': [
        {'rows': 16, 'cols': 32, 'defaultInstrument': 'glass'},
        {'rows': 16, 'cols': 32, 'defaultInstrument': 'fm'},
        {'rows': 16, 'cols': 32, 'defaultInstrument': 'sub'},
        {'rows': 16, 'cols': 32, 'defaultInstrument': 'dust'},
    ],
    'mainGridLayout': {'cols': 2},
    'melodyGrids': [
        {'rows': 8, 'cols': 32},
        {'rows': 8, 'cols': 32},
    ],
    'melodyDefaultInstrument': 'pad',
    'effects': {
        'reverbWet': 0.22,
        'reverbDark': 0.8,
        'reverbLength': 3.5,
        'delayL': 0.45,
        'delayR': 0.30,
        'delayFeedback': 0.4,
        'delayDarkLP': 1200,
        'delayWet': 0.24,
        'compThreshold': -12,
        'compRatio': 8,
    },
    'numPatterns': 8,
    'defaultMelN': 2,
    'cellSize': 16,
    'fx': FX_CONFIG,
}


def generate(out_dir, sr=SR, fmt='ogg'):
    instruments = []

    for inst_id, gen_fn in INSTRUMENTS.items():
        freqs = MEL_FREQS if inst_id in MEL_INSTS else FREQS
        bits, down, tape_drive = CRUSH[inst_id]
        fi_ms, fo_ms = FADE[inst_id]
        is_mel = inst_id in MEL_INSTS

        print(f'  {inst_id}: {len(freqs)} pitches')
        segments = []

        for i, freq in enumerate(freqs):
            sig = gen_fn(freq, sr)

            if tape_drive > 1:
                for ch in range(sig.shape[1] if sig.ndim > 1 else 1):
                    col = sig[:, ch] if sig.ndim > 1 else sig
                    x = col * tape_drive
                    col[:] = np.where(x > 0, np.tanh(x * 1.2) / 1.2, np.tanh(x * 0.9) / 0.9)
            if down > 1 or bits < 16:
                for ch in range(sig.shape[1] if sig.ndim > 1 else 1):
                    col = sig[:, ch] if sig.ndim > 1 else sig
                    col[:] = bitcrush(col, bits, down)

            sig = normalize(sig, 0.85)
            fade_in_ms = 8 if is_mel else fi_ms
            fade_out_ms = 80 if is_mel else fo_ms
            for ch in range(sig.shape[1] if sig.ndim > 1 else 1):
                col = sig[:, ch] if sig.ndim > 1 else sig
                fade_in(col, fade_in_ms, sr)
                fade_out(col, fade_out_ms, sr)

            segments.append(sig)

        entry = {
            'id': inst_id,
            'label': LABELS[inst_id],
            'pitchCount': len(freqs),
            'type': 'melody' if is_mel else 'main',
        }
        entry.update(write_sprite(out_dir, inst_id, segments, sr, fmt))
        instruments.append(entry)

    ir = generate_reverb_ir(3.5, dark=0.8, sr=sr)
    export_ogg(ir, os.path.join(out_dir, f'reverb_ir.{fmt}'), sr)

    config = dict(CONFIG)
    config['mainFreqs'] = FREQS
    config['melodyFreqs'] = MEL_FREQS
    write_manifest(out_dir, config, instruments, fmt=fmt)
