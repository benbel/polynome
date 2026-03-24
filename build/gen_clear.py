"""Generate clear mode audio assets.

Early electronic / Kraftwerk-to-Warp — precise, crystalline, mechanical.
Sharp envelopes, clean synthesis, one warm element (strings) against cold backdrop.
"""

import os
import numpy as np
from common import (
    SR, sine, saw, pulse, noise,
    env_adsr, env_exp_decay, lowpass, highpass, bandpass,
    moog_ladder, tanh_saturate, comb_filter,
    normalize, fade_in, fade_out, to_stereo, mix_stereo,
    export_ogg, write_manifest, generate_reverb_ir,
)

FREQS = [523, 440, 370, 311, 262, 220, 175, 147, 123, 104, 82, 65, 55, 44, 33, 25]
MEL_FREQS = [659, 523, 440, 349, 262, 220, 175, 131]

DURATIONS = {
    'moog_bass': 2.5, 'pluck': 1.8, 'string_machine': 3.0,
    'sync': 2.0, 'bell': 3.0, 'wurli': 2.5, 'clav': 1.5, 'celesta': 2.5,
}


def gen_moog_bass(freq, sr=SR):
    f = freq
    dur = DURATIONS['moog_bass']
    n = int(sr * dur)

    sig = saw(f, dur, sr)

    # Filter envelope: bright attack sweeping down — restored squelch
    cutoff = np.full(n, f * 0.6)
    env_n = int(sr * 0.4)
    cutoff[:env_n] = np.linspace(800, 80, env_n)
    cutoff[env_n:] = 80

    sig = moog_ladder(sig, cutoff, resonance=1.8, sr=sr)

    # Sub sine one octave down
    sub = sine(f * 0.5, dur, sr) * 0.3

    mix = sig * 0.7 + sub
    env = env_exp_decay(dur, 5, 0.8, sr)
    mix *= env

    result = normalize(mix, 0.85)
    fade_in(result, 5, sr)
    fade_out(result, 40, sr)
    return to_stereo(result, 0)


def gen_pluck(freq, sr=SR):
    """Waveguide pluck — percussive mid-register, clean metallic string."""
    f = freq
    dur = DURATIONS['pluck']
    n = int(sr * dur)

    # Noise excitation shaped for bright attack
    exc_dur = 0.006
    exc = noise(exc_dur, sr) * 0.9
    exc = bandpass(exc, max(20, f * 0.8), min(sr / 2 - 100, f * 6), sr)
    exc_pad = np.zeros(n)
    exc_pad[:len(exc)] = exc

    # Primary string resonance
    delay1 = max(1, int(sr / f))
    c1 = comb_filter(exc_pad, delay1, feedback=0.92, lp_freq=min(f * 5, sr / 2 - 100), sr=sr)

    # Slight detuned second string for chorus
    delay2 = max(1, int(sr / (f * 1.003)))
    c2 = comb_filter(exc_pad, delay2, feedback=0.90, lp_freq=min(f * 4, sr / 2 - 100), sr=sr)

    mix = c1 * 0.6 + c2 * 0.4

    # Bright attack transient
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

    # 8 detuned saw pairs
    voices = []
    for i in range(8):
        detune = (i - 3.5) * 8
        inst_freq = f * (2 ** (detune / 1200))
        phase = np.cumsum(np.full(n, inst_freq) / sr)
        voice = 2 * (phase % 1) - 1
        pan = (i - 3.5) / 4
        voices.append(to_stereo(voice * 0.06, pan))

    mix = mix_stereo(*voices)[:n]

    # Frequency-relative cutoff instead of fixed 4kHz
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
    """FM bell with longer decay and more inharmonic partials."""
    f = freq
    dur = DURATIONS['bell']
    n = int(sr * dur)
    t = np.arange(n) / sr

    # Primary FM: carrier:modulator = 1:3.5
    mod_index = np.linspace(5, 0.1, n)
    modulator = np.sin(2 * np.pi * f * 3.5 * t) * mod_index * f
    carrier = np.sin(2 * np.pi * f * t + modulator / f)

    # Secondary inharmonic partial cloud (1:5.3 and 1:7.1 ratios)
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
    """Celesta — inharmonic partial cloud with staggered decays (metal plate model)."""
    f = freq
    dur = DURATIONS['celesta']
    n = int(sr * dur)
    t = np.arange(n) / sr

    # Inharmonic partial series (metal plate ratios, not integer harmonics)
    partials = [
        (1.0,   0.50, 0.5),   # fundamental
        (2.76,  0.35, 0.35),   # ~minor 7th + octave
        (3.01,  0.25, 0.3),    # slightly sharp 12th
        (5.40,  0.18, 0.2),    # high inharmonic
        (8.93,  0.10, 0.12),   # shimmer
        (13.2,  0.05, 0.08),   # sparkle
    ]

    sig = np.zeros(n)
    for ratio, amp, decay_time in partials:
        pf = f * ratio
        if pf >= sr / 2:
            continue
        partial = np.sin(2 * np.pi * pf * t) * amp
        p_env = env_exp_decay(dur, 1, decay_time, sr)
        sig += partial * p_env

    # Hammer click
    click = noise(0.002, sr)
    click = highpass(click, min(3000, sr / 2 - 100), sr) * 0.12
    sig[:len(click)] += click[:min(len(click), n)]

    result = normalize(sig, 0.85)
    fade_in(result, 1, sr)
    fade_out(result, 50, sr)
    return to_stereo(result, 0)


INSTRUMENTS = {
    'moog_bass': gen_moog_bass, 'pluck': gen_pluck,
    'string_machine': gen_string_machine, 'sync': gen_sync, 'bell': gen_bell,
    'wurli': gen_wurli, 'clav': gen_clav, 'celesta': gen_celesta,
}
MEL_INSTS = {'wurli', 'clav', 'celesta'}

FX_CONFIG = {
    'moog_bass': {'delay': 0.08, 'reverb': 0.12, 'gain': 0.30},
    'pluck': {'delay': 0.10, 'reverb': 0.12, 'gain': 0.26},
    'string_machine': {'delay': 0.20, 'reverb': 0.35, 'gain': 0.18},
    'sync': {'delay': 0.12, 'reverb': 0.18, 'gain': 0.22},
    'bell': {'delay': 0.30, 'reverb': 0.45, 'gain': 0.16},
    'wurli': {'delay': 0.18, 'reverb': 0.25, 'gain': 0.20},
    'clav': {'delay': 0.10, 'reverb': 0.15, 'gain': 0.24},
    'celesta': {'delay': 0.25, 'reverb': 0.40, 'gain': 0.18},
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

    ir = generate_reverb_ir(2.5, dark=0.4, sr=sr)
    export_ogg(ir, os.path.join(out_dir, 'reverb_ir.ogg'), sr)
    write_manifest(out_dir, manifest_insts, FX_CONFIG)
