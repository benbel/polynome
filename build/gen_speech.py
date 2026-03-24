"""Generate speech mode audio assets.

Phoneme-driven percussive speech. Uses pure formant synthesis
(no external TTS dependency) to generate phoneme-like sounds.
Each phoneme class becomes an instrument with pitch variants.
"""

import os
import numpy as np
from common import (
    SR, sine, noise,
    env_adsr, env_exp_decay, lowpass, highpass, bandpass,
    normalize, fade_in, fade_out,
    to_stereo, export_ogg, write_manifest, generate_reverb_ir,
)

FREQS = [523, 440, 349, 294, 220, 175, 131, 110]
MEL_FREQS = [659, 523, 440, 349, 262, 220, 175, 131]

DURATIONS = {
    'vowels': 0.45, 'clicks': 0.20, 'hiss': 0.30,
    'nasal': 0.40, 'liquid': 0.35, 'syllables': 0.65,
}


def glottal_source(f0, dur, sr=SR):
    """Simple glottal source for speech synthesis."""
    n = int(sr * dur)
    phase = np.cumsum(np.full(n, f0) / sr)
    cycle_pos = phase % 1
    oq = 0.6
    pulse = np.where(cycle_pos < oq, 3 * (cycle_pos / oq) ** 2 - 2 * (cycle_pos / oq) ** 3, 0)
    return np.diff(pulse, prepend=0)


def formant_synth(source, formants, sr=SR):
    """Apply formant filter bank. formants: list of (freq, bw, gain_db)."""
    out = np.zeros_like(source)
    for f, bw, gain in formants:
        low = max(20, f - bw / 2)
        high = min(sr / 2 - 100, f + bw / 2)
        if low >= high:
            continue
        filtered = bandpass(source, low, high, sr, order=2)
        out += filtered * (10 ** (gain / 20))
    return out


# ======================== INSTRUMENTS ========================

def gen_vowels(freq, sr=SR):
    """Tonal vowel sounds — pitched, with formant transition for naturalness."""
    dur = DURATIONS['vowels']
    n = int(sr * dur)

    # Vowel formants with F1-F3 (speech-realistic values)
    vowels = [
        [(730, 90, 0), (1090, 110, -6), (2440, 170, -12)],   # a
        [(270, 60, 0), (2290, 200, -6), (3010, 200, -12)],    # i
        [(300, 60, 0), (870, 100, -6), (2240, 170, -12)],     # u
        [(530, 80, 0), (1840, 150, -6), (2480, 180, -12)],    # e
        [(570, 80, 0), (840, 100, -6), (2410, 170, -12)],     # o
    ]
    vidx = int(freq) % len(vowels)
    source = glottal_source(freq, dur, sr)

    # Formant transition: start from neutral schwa, glide into target vowel
    schwa = [(500, 80, 0), (1500, 120, -6), (2500, 200, -14)]
    target = vowels[vidx]
    # Crossfade: first 30% is transition, rest is steady
    trans_n = int(n * 0.3)
    sig_schwa = formant_synth(source, schwa, sr)[:n]
    sig_target = formant_synth(source, target, sr)[:n]
    xfade = np.linspace(0, 1, trans_n)
    sig = sig_target.copy()
    sig[:trans_n] = sig_schwa[:trans_n] * (1 - xfade) + sig_target[:trans_n] * xfade

    sig = sig[:n]
    env = env_adsr(dur, 0.008, 0.06, 0.8, 0.12, sr)
    sig *= env[:len(sig)]

    result = normalize(sig, 0.85)
    fade_in(result, 3, sr)
    fade_out(result, 30, sr)
    return to_stereo(result, 0)


def gen_clicks(freq, sr=SR):
    """Percussive plosive sounds — p/t/k-like bursts with aspiration."""
    dur = DURATIONS['clicks']
    n = int(sr * dur)

    # Plosive: silence → burst → brief aspiration → voiced tail
    # Burst: very short noise shaped by place of articulation
    burst_dur = 0.006
    burst = noise(burst_dur, sr)
    # Place of articulation varies by pitch: labial→alveolar→velar
    place = int(freq) % 3
    if place == 0:    # labial (p/b) — low burst
        burst = bandpass(burst, 200, 800, sr)
    elif place == 1:  # alveolar (t/d) — mid-high burst
        burst = bandpass(burst, 2000, min(sr/2-100, 5000), sr)
    else:             # velar (k/g) — mid burst
        burst = bandpass(burst, 800, 2000, sr)

    # Aspiration (breathy noise after burst)
    asp = noise(0.025, sr) * 0.4
    asp = bandpass(asp, 1000, min(sr/2-100, 4000), sr)
    asp_env = env_exp_decay(0.025, 0.2, 0.015, sr)
    asp *= asp_env

    # Short voiced tail at pitch frequency
    tail = sine(freq, 0.04, sr) * 0.2
    tail_env = env_exp_decay(0.04, 0.5, 0.025, sr)
    tail *= tail_env

    sig = np.zeros(n)
    sig[:len(burst)] = burst
    asp_start = len(burst)
    sig[asp_start:asp_start+len(asp)] += asp
    tail_start = asp_start + int(sr * 0.01)
    end = min(tail_start + len(tail), n)
    sig[tail_start:end] += tail[:end-tail_start]

    result = normalize(sig, 0.85)
    return to_stereo(result, 0)


def gen_hiss(freq, sr=SR):
    """Fricative sounds — s/sh/f-like with spectral variation."""
    dur = DURATIONS['hiss']
    n = int(sr * dur)

    sig = noise(dur, sr)

    # Different fricative type by pitch: /f/ → /s/ → /ʃ/
    ftype = int(freq) % 3
    if ftype == 0:    # /f/ — broad, low-energy
        sig = bandpass(sig, 1500, min(sr/2-100, 6000), sr) * 0.5
    elif ftype == 1:  # /s/ — sharp, high center
        sig = bandpass(sig, 4000, min(sr/2-100, 9000), sr) * 0.7
    else:             # /ʃ/ — broader, lower than /s/
        sig = bandpass(sig, 2500, min(sr/2-100, 7000), sr) * 0.6

    env = env_adsr(dur, 0.005, 0.03, 0.7, 0.05, sr)
    sig *= env[:len(sig)]

    result = normalize(sig, 0.85)
    fade_in(result, 2, sr)
    fade_out(result, 10, sr)
    return to_stereo(result, 0)


def gen_nasal(freq, sr=SR):
    """Nasal sounds — m/n/ng-like with nasal pole-zero pairs."""
    dur = DURATIONS['nasal']
    n = int(sr * dur)

    source = glottal_source(freq, dur, sr)

    # Nasal type by pitch: /m/ → /n/ → /ŋ/
    ntype = int(freq) % 3
    if ntype == 0:    # /m/ — low nasal pole
        pole_freq, zero_freq = 250, 800
    elif ntype == 1:  # /n/ — mid nasal pole
        pole_freq, zero_freq = 350, 1200
    else:             # /ŋ/ — higher nasal pole
        pole_freq, zero_freq = 400, 1500

    sig = lowpass(source, 600, sr)
    # Nasal pole (resonance)
    nasal = bandpass(source, max(20, pole_freq - 80), min(sr/2-100, pole_freq + 80), sr) * 0.6
    sig = sig[:n] + nasal[:n]
    # Anti-formant (zero — removes oral resonance)
    antif = bandpass(source, max(20, zero_freq - 100), min(sr/2-100, zero_freq + 100), sr) * 0.4
    sig -= antif[:n]

    env = env_adsr(dur, 0.01, 0.06, 0.8, 0.08, sr)
    sig *= env[:len(sig)]

    result = normalize(sig, 0.85)
    fade_in(result, 3, sr)
    fade_out(result, 15, sr)
    return to_stereo(result, 0)


def gen_liquid(freq, sr=SR):
    """Liquid sounds — l/r with formant glide for flowing character."""
    dur = DURATIONS['liquid']
    n = int(sr * dur)

    source = glottal_source(freq, dur, sr)

    # /l/ vs /r/ by pitch
    is_r = int(freq) % 2 == 0
    if is_r:
        # /r/ — F3 dip is characteristic
        formants = [(500, 100, 0), (1100, 150, -4), (1600, 200, -10), (2800, 250, -16)]
    else:
        # /l/ — F2/F3 closer together
        formants = [(350, 80, 0), (1100, 130, -4), (2800, 200, -10)]

    # Quick formant transition from neutral into liquid
    neutral = [(500, 100, 0), (1500, 150, -6), (2500, 200, -14)]
    trans_n = int(n * 0.25)
    sig_n = formant_synth(source, neutral, sr)[:n]
    sig_t = formant_synth(source, formants, sr)[:n]
    xfade = np.linspace(0, 1, trans_n)
    sig = sig_t.copy()
    sig[:trans_n] = sig_n[:trans_n] * (1 - xfade) + sig_t[:trans_n] * xfade

    env = env_adsr(dur, 0.01, 0.06, 0.65, 0.06, sr)
    sig *= env[:len(sig)]

    result = normalize(sig[:n], 0.85)
    fade_in(result, 3, sr)
    fade_out(result, 15, sr)
    return to_stereo(result, 0)


def _make_syllable(sig, offset, freq, bw_lo, bw_hi, bdur, asp, is_nasal,
                    vowel_formants, vowel_dur, sr):
    """Render one CV syllable into sig at the given sample offset."""
    n = len(sig)

    # Consonant onset
    if is_nasal:
        nasal_dur = 0.025
        ns = glottal_source(freq, nasal_dur, sr)
        ns = lowpass(ns, 400, sr) * 0.5
        end = min(offset + len(ns), n)
        sig[offset:end] += ns[:end - offset]
    else:
        burst = noise(bdur, sr)
        burst = bandpass(burst, max(20, bw_lo), min(sr/2-100, bw_hi), sr)
        burst_env = env_exp_decay(bdur, 0.2, 0.004, sr)
        burst *= burst_env
        end = min(offset + len(burst), n)
        sig[offset:end] += burst[:end - offset]

    # Aspiration
    if asp > 0:
        asp_sig = noise(0.02, sr) * asp
        asp_sig = bandpass(asp_sig, 1500, min(sr/2-100, 5000), sr)
        asp_env = env_exp_decay(0.02, 0.2, 0.012, sr)
        asp_sig *= asp_env
        asp_start = offset + int(sr * bdur)
        end = min(asp_start + len(asp_sig), n)
        if end > asp_start:
            sig[asp_start:end] += asp_sig[:end - asp_start]

    # Vowel body
    source = glottal_source(freq, vowel_dur, sr)
    vowel = formant_synth(source, vowel_formants, sr)
    vowel_start = offset + int(sr * 0.02)
    vowel_ramp = int(sr * 0.015)
    end = min(vowel_start + len(vowel), n)
    if end > vowel_start:
        vslice = vowel[:end - vowel_start]
        ramp = np.ones(len(vslice))
        ramp[:min(vowel_ramp, len(ramp))] = np.linspace(0, 1, min(vowel_ramp, len(ramp)))
        sig[vowel_start:end] += vslice * ramp


def gen_syllables(freq, sr=SR):
    """Multi-syllable word-like sounds — CVCV patterns so music emerges from talk."""
    dur = DURATIONS['syllables']
    n = int(sr * dur)

    # 8 word-like patterns for variety: baba, tiku, meno, laba, kume, nola, bati, mela
    consonant_configs = [
        (200, 800, 0.006, 0.15, False),    # b — labial stop
        (2000, 5000, 0.004, 0.35, False),   # t — alveolar stop
        (800, 2000, 0.005, 0.25, False),    # k — velar stop
        (200, 800, 0.006, 0.0, True),       # m — nasal
        (200, 500, 0.008, 0.0, True),       # n — nasal
        (300, 1500, 0.003, 0.1, False),     # l — lateral
    ]
    vowel_sets = [
        [(730, 90, 0), (1090, 110, -6), (2440, 170, -12)],  # a
        [(270, 60, 0), (2290, 200, -6), (3010, 200, -12)],  # i
        [(300, 60, 0), (870, 100, -6), (2240, 170, -12)],   # u
        [(530, 80, 0), (1840, 150, -6), (2480, 180, -12)],  # e
        [(570, 80, 0), (840, 100, -6), (2410, 170, -12)],   # o
    ]
    # Word patterns: pairs of (consonant_idx, vowel_idx)
    words = [
        [(0, 0), (0, 0)],  # ba-ba
        [(1, 3), (2, 2)],  # te-ku
        [(3, 3), (4, 4)],  # me-no
        [(5, 0), (0, 0)],  # la-ba
        [(2, 2), (3, 3)],  # ku-me
        [(4, 4), (5, 0)],  # no-la
        [(0, 0), (1, 3)],  # ba-te
        [(3, 3), (5, 0)],  # me-la
    ]

    wtype = int(freq) % len(words)
    word = words[wtype]
    sig = np.zeros(n)
    syl_dur = dur / len(word)

    for si, (ci, vi) in enumerate(word):
        offset = int(sr * si * syl_dur)
        bw_lo, bw_hi, bdur, asp, is_nasal = consonant_configs[ci]
        _make_syllable(sig, offset, freq, bw_lo, bw_hi, bdur, asp, is_nasal,
                       vowel_sets[vi], syl_dur, sr)

    env = env_adsr(dur, 0.005, 0.06, 0.7, 0.12, sr)
    sig *= env[:len(sig)]

    result = normalize(sig, 0.85)
    fade_in(result, 2, sr)
    fade_out(result, 30, sr)
    return to_stereo(result, 0)


INSTRUMENTS = {
    'vowels': gen_vowels, 'clicks': gen_clicks, 'hiss': gen_hiss,
    'nasal': gen_nasal, 'liquid': gen_liquid, 'syllables': gen_syllables,
}
MEL_INSTS = {'syllables'}

FX_CONFIG = {
    'vowels': {'delay': 0.15, 'reverb': 0.25, 'gain': 0.24},
    'clicks': {'delay': 0.08, 'reverb': 0.12, 'gain': 0.30},
    'hiss': {'delay': 0.20, 'reverb': 0.30, 'gain': 0.20},
    'nasal': {'delay': 0.12, 'reverb': 0.20, 'gain': 0.26},
    'liquid': {'delay': 0.18, 'reverb': 0.28, 'gain': 0.22},
    'syllables': {'delay': 0.15, 'reverb': 0.25, 'gain': 0.22},
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

    ir = generate_reverb_ir(2.0, dark=0.5, sr=sr)
    export_ogg(ir, os.path.join(out_dir, 'reverb_ir.ogg'), sr)
    write_manifest(out_dir, manifest_insts, FX_CONFIG)
