"""Generate clear mode audio assets.

Crystalline analog synthesis inspired by Wendy Carlos.
Moog ladder filter, precise clean synthesis, microtonal options.
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

FREQS = [262, 220, 196, 165, 131, 110, 98, 82, 73, 65, 55, 49, 41, 33, 27, 21]
MEL_FREQS = [523, 440, 392, 330, 262, 220, 196, 165]

DURATIONS = {
    'moog_bass': 2.5, 'moog_lead': 2.0, 'string_machine': 3.0,
    'sync': 2.0, 'bell': 3.0, 'wurli': 2.5, 'clav': 1.5, 'celesta': 2.0,
}


def gen_moog_bass(freq, sr=SR):
    f = freq
    dur = DURATIONS['moog_bass']
    n = int(sr * dur)

    # Single saw through Moog ladder
    sig = saw(f, dur, sr)

    # Filter envelope: 800Hz -> 80Hz over 400ms
    cutoff = np.full(n, f * 0.6)
    env_n = int(sr * 0.4)
    cutoff[:env_n] = np.linspace(800, 80, env_n)
    cutoff[env_n:] = 80

    sig = moog_ladder(sig, cutoff, resonance=2.5, sr=sr)

    # Sub sine one octave down
    sub = sine(f * 0.5, dur, sr) * 0.3

    mix = sig * 0.7 + sub
    env = env_exp_decay(dur, 5, 0.8, sr)
    mix *= env

    result = normalize(mix, 0.85)
    fade_in(result, 5, sr)
    fade_out(result, 40, sr)
    return to_stereo(result, 0)


def gen_moog_lead(freq, sr=SR):
    f = freq
    dur = DURATIONS['moog_lead']
    n = int(sr * dur)
    t = np.arange(n) / sr

    # Pulse wave with PWM
    pwm = 0.2 + 0.3 * np.sin(2 * np.pi * 2 * t)  # LFO 2Hz on pulse width
    phase = np.cumsum(np.full(n, f) / sr)
    sig = np.where(phase % 1 < pwm, 1.0, -1.0)

    # Moog ladder
    cutoff = np.full(n, f * 3)
    env_n = int(sr * 0.1)
    cutoff[:env_n] = np.linspace(f * 8, f * 3, env_n)
    sig = moog_ladder(sig, cutoff, resonance=1.5, sr=sr)

    env = env_exp_decay(dur, 3, 0.3, sr)
    sig *= env

    result = normalize(sig, 0.85)
    fade_in(result, 3, sr)
    fade_out(result, 30, sr)
    return to_stereo(result, 0)


def gen_string_machine(freq, sr=SR):
    f = freq
    dur = DURATIONS['string_machine']
    n = int(sr * dur)
    t = np.arange(n) / sr

    # 8 detuned saw pairs
    voices = []
    for i in range(8):
        detune = (i - 3.5) * 8  # ±8 cents spread
        inst_freq = f * (2 ** (detune / 1200))
        phase = np.cumsum(np.full(n, inst_freq) / sr)
        voice = 2 * (phase % 1) - 1
        pan = (i - 3.5) / 4
        voices.append(to_stereo(voice * 0.06, pan))

    mix = mix_stereo(*voices)[:n]

    # Ensemble chorus (simple delay modulation)
    for ch in range(2):
        mix[:, ch] = lowpass(mix[:, ch], 4000, sr)

    env = env_adsr(dur, 0.2, 0.1, 0.8, 0.5, sr)
    mix *= env[:, np.newaxis]

    return normalize(mix, 0.85)


def gen_sync(freq, sr=SR):
    f = freq
    dur = DURATIONS['sync']
    n = int(sr * dur)
    t = np.arange(n) / sr

    # Hard sync: master at f, slave at f*ratio (ratio sweeps via LFO)
    ratio = 2.5 + 0.5 * np.sin(2 * np.pi * 0.8 * t)
    master_phase = np.cumsum(np.full(n, f) / sr)
    slave_freq = f * ratio
    slave_phase = np.cumsum(slave_freq / sr)

    # Reset slave phase on master zero-crossings
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

    # 2-operator FM: carrier:modulator = 1:3.5
    mod_index = np.linspace(5, 0.1, n)
    modulator = np.sin(2 * np.pi * f * 3.5 * t) * mod_index * f
    carrier = np.sin(2 * np.pi * f * t + modulator / f)

    env = env_exp_decay(dur, 2, 0.6, sr)
    sig = carrier * env

    result = normalize(sig, 0.85)
    fade_in(result, 2, sr)
    fade_out(result, 50, sr)
    return to_stereo(result, 0)


def gen_wurli(freq, sr=SR):
    f = freq
    dur = DURATIONS['wurli']
    n = int(sr * dur)
    t = np.arange(n) / sr

    # FM ratio 1:1, index envelope 3->0.2
    mod_index = np.linspace(3, 0.2, n) * f
    modulator = np.sin(2 * np.pi * f * t)
    sig = np.sin(2 * np.pi * f * t + modulator * mod_index / f)

    # Slight tremolo
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

    # Short noise impulse -> high-feedback comb
    exc = noise(0.003, sr)
    exc = highpass(exc, 1000, sr) * 0.8
    exc_pad = np.zeros(n)
    exc_pad[:len(exc)] = exc

    delay = max(1, int(sr / f))
    sig = comb_filter(exc_pad, delay, feedback=0.9, lp_freq=f * 5, sr=sr)

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

    # Sine + slightly inharmonic partial (3.01x)
    sig = sine(f, dur, sr) * 0.7 + sine(f * 3.01, dur, sr) * 0.3

    env = env_exp_decay(dur, 1, 0.2, sr)
    sig *= env

    result = normalize(sig, 0.85)
    fade_in(result, 1, sr)
    fade_out(result, 30, sr)
    return to_stereo(result, 0)


INSTRUMENTS = {
    'moog_bass': gen_moog_bass, 'moog_lead': gen_moog_lead,
    'string_machine': gen_string_machine, 'sync': gen_sync, 'bell': gen_bell,
    'wurli': gen_wurli, 'clav': gen_clav, 'celesta': gen_celesta,
}
MEL_INSTS = {'wurli', 'clav', 'celesta'}

FX_CONFIG = {
    'moog_bass': {'delay': 0.08, 'reverb': 0.12, 'gain': 0.30},
    'moog_lead': {'delay': 0.15, 'reverb': 0.20, 'gain': 0.24},
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
