import numpy as np
from scipy.signal import sosfilt, butter, lfilter
from scipy.io import wavfile
import soundfile
import json, os

SR = 44100


def sine(freq, dur, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t)


def saw(freq, dur, sr=SR, num_harmonics=30):
    t = np.arange(int(sr * dur)) / sr
    out = np.zeros_like(t)
    for k in range(1, num_harmonics + 1):
        if k * freq > sr / 2:
            break
        out += ((-1) ** (k + 1)) * np.sin(2 * np.pi * k * freq * t) / k
    return out * (2 / np.pi)


def pulse(freq, dur, pw=0.5, sr=SR, num_harmonics=30):
    t = np.arange(int(sr * dur)) / sr
    out = np.zeros_like(t)
    for k in range(1, num_harmonics + 1):
        if k * freq > sr / 2:
            break
        out += np.sin(np.pi * k * pw) * np.sin(2 * np.pi * k * freq * t) / k
    return out * (4 / np.pi)


def noise(dur, sr=SR):
    return np.random.uniform(-1, 1, int(sr * dur))


def stereo_noise(dur, sr=SR):
    return np.column_stack([noise(dur, sr), noise(dur, sr)])


def env_adsr(dur, attack, decay, sustain, release, sr=SR):
    n = int(sr * dur)
    a = int(sr * attack)
    d = int(sr * decay)
    r = int(sr * release)
    s = max(0, n - a - d - r)
    env = np.concatenate([
        np.linspace(0, 1, max(a, 1)),
        np.linspace(1, sustain, max(d, 1)),
        np.full(s, sustain),
        np.linspace(sustain, 0, max(r, 1)),
    ])
    return env[:n]


def env_exp_decay(dur, attack_ms, decay_time, sr=SR):
    n = int(sr * dur)
    a = int(sr * attack_ms / 1000)
    env = np.zeros(n)
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    t = np.arange(n - a) / sr
    env[a:] = np.exp(-t / max(decay_time, 0.001))
    return env


def lowpass(sig, freq, sr=SR, order=4):
    freq = np.clip(freq, 20, sr / 2 - 1)
    sos = butter(order, freq, btype='low', fs=sr, output='sos')
    return sosfilt(sos, sig)


def highpass(sig, freq, sr=SR, order=4):
    freq = np.clip(freq, 20, sr / 2 - 1)
    sos = butter(order, freq, btype='high', fs=sr, output='sos')
    return sosfilt(sos, sig)


def bandpass(sig, low, high, sr=SR, order=2):
    low = np.clip(low, 20, sr / 2 - 1)
    high = np.clip(high, low + 1, sr / 2 - 1)
    sos = butter(order, [low, high], btype='band', fs=sr, output='sos')
    return sosfilt(sos, sig)


def _shelf_coeffs(freq, gain_db, sr, high):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * np.clip(freq, 20, sr / 2 - 1) / sr
    c = np.cos(w0)
    beta = 2 * np.sqrt(A) * (np.sin(w0) / 2 * np.sqrt(2))

    if high:
        b = [A * ((A + 1) + (A - 1) * c + beta),
             -2 * A * ((A - 1) + (A + 1) * c),
             A * ((A + 1) + (A - 1) * c - beta)]
        a = [(A + 1) - (A - 1) * c + beta,
             2 * ((A - 1) - (A + 1) * c),
             (A + 1) - (A - 1) * c - beta]
    else:
        b = [A * ((A + 1) - (A - 1) * c + beta),
             2 * A * ((A - 1) - (A + 1) * c),
             A * ((A + 1) - (A - 1) * c - beta)]
        a = [(A + 1) + (A - 1) * c + beta,
             -2 * ((A - 1) + (A + 1) * c),
             (A + 1) + (A - 1) * c - beta]

    return np.array(b) / a[0], np.array(a) / a[0]


def high_shelf(sig, freq, gain_db, sr=SR):
    if gain_db == 0:
        return sig
    b, a = _shelf_coeffs(freq, gain_db, sr, True)
    return lfilter(b, a, sig)


def low_shelf(sig, freq, gain_db, sr=SR):
    if gain_db == 0:
        return sig
    b, a = _shelf_coeffs(freq, gain_db, sr, False)
    return lfilter(b, a, sig)


def filter_sweep(sig, start_freq, end_freq, sr=SR, order=4, block_size=256):
    n = len(sig)
    out = np.zeros(n)
    freqs = np.logspace(np.log10(max(start_freq, 20)), np.log10(min(end_freq, sr / 2 - 100)),
                        n // block_size + 1)
    freqs = np.clip(freqs, 20, sr / 2 - 100)
    zi = None
    for i, fc in enumerate(freqs):
        start = i * block_size
        end = min(start + block_size, n)
        if start >= n:
            break
        sos = butter(order, fc, btype='low', fs=sr, output='sos')
        if zi is None:
            block, zi = sosfilt(sos, sig[start:end], zi=np.zeros((sos.shape[0], 2)))
        else:
            block, zi = sosfilt(sos, sig[start:end], zi=zi)
        out[start:end] = block
    return out


def moog_ladder(sig, cutoff_hz, resonance, sr=SR):
    if isinstance(cutoff_hz, (int, float)):
        cutoff_hz = np.full(len(sig), cutoff_hz)
    s = [0.0, 0.0, 0.0, 0.0]
    out = np.zeros(len(sig))
    for i in range(len(sig)):
        g = 1 - np.exp(-2 * np.pi * cutoff_hz[i] / sr)
        feedback = resonance * np.tanh(s[3])
        x = np.tanh(sig[i] - feedback)
        for stage in range(4):
            s[stage] += g * (x - s[stage])
            x = s[stage]
        out[i] = s[3]
    return out


def wavefold(sig, folds=3):
    x = sig * folds
    return (4 / np.pi) * np.arcsin(np.sin(np.pi * x / 2))


def tanh_saturate(sig, drive=4):
    return np.tanh(sig * drive) / np.tanh(drive)


def asymmetric_saturate(sig, drive=2, asymmetry=0.15):
    pos = np.tanh(sig * drive * (1 + asymmetry)) / (1 + asymmetry)
    neg = np.tanh(sig * drive * (1 - asymmetry)) / (1 - asymmetry)
    return np.where(sig >= 0, pos, neg)


def bitcrush(sig, bits=10, downsample=3):
    step = 2 ** (-(bits - 1))
    crushed = np.round(sig / step) * step
    if downsample > 1:
        held = crushed.copy()
        for i in range(len(held)):
            if i % downsample != 0:
                held[i] = held[i - 1]
        return held
    return crushed


def comb_filter(sig, delay_samples, feedback=0.7, lp_freq=None, sr=SR):
    n = len(sig)
    out = np.zeros(n)
    buf = np.zeros(delay_samples)
    p = 0
    for i in range(n):
        out[i] = sig[i] + buf[p] * feedback
        buf[p] = out[i]
        if lp_freq and p > 0:
            alpha = 1 - np.exp(-2 * np.pi * lp_freq / sr)
            buf[p] = alpha * buf[p] + (1 - alpha) * buf[(p - 1) % delay_samples]
        p = (p + 1) % delay_samples
    return out


def normalize(sig, peak=0.85):
    mx = np.max(np.abs(sig))
    if mx > 0:
        sig = sig * (peak / mx)
    return sig


def _ramp(n, start, stop, ndim):
    ramp = np.linspace(start, stop, n)
    return ramp[:, np.newaxis] if ndim == 2 else ramp


def fade_in(sig, ms, sr=SR):
    n = min(int(sr * ms / 1000), len(sig))
    if n > 0:
        sig[:n] *= _ramp(n, 0, 1, sig.ndim)
    return sig


def fade_out(sig, ms, sr=SR):
    n = min(int(sr * ms / 1000), len(sig))
    if n > 0:
        sig[-n:] *= _ramp(n, 1, 0, sig.ndim)
    return sig


def to_stereo(mono, pan=0):
    l = mono * np.sqrt((1 - pan) / 2)
    r = mono * np.sqrt((1 + pan) / 2)
    return np.column_stack([l, r])


def mix_stereo(*signals):
    maxlen = max(len(s) for s in signals)
    out = np.zeros((maxlen, 2))
    for s in signals:
        if s.ndim == 1:
            s = to_stereo(s, 0)
        out[:len(s)] += s
    return out


WRITE_BLOCK = 44100


# Safari -- desktop and every iPhone -- has no Ogg Vorbis decoder, so an ogg
# build is silent on iOS even though the page loads and animates normally.
# MP3 is the lossy format every browser we target decodes.
def export_audio(stereo_signal, path, sr=SR):
    signal = np.ascontiguousarray(np.clip(stereo_signal, -1, 1), dtype='float32')
    ext = os.path.splitext(path)[1].lower()

    if ext == '.wav':
        wavfile.write(path, sr, (signal * 32767).astype(np.int16))
        return
    if ext != '.mp3':
        raise ValueError(f'{ext} is not a format browsers all decode: use .mp3 or .wav')

    channels = signal.shape[1] if signal.ndim > 1 else 1
    with soundfile.SoundFile(path, 'w', samplerate=sr, channels=channels,
                             format='MP3', subtype='MPEG_LAYER_III') as f:
        for i in range(0, len(signal), WRITE_BLOCK):
            f.write(signal[i:i + WRITE_BLOCK])


SPRITE_LEAD = 0.005
SPRITE_GAP = 0.2


def write_sprite(out_dir, inst_id, segments, sr=SR, fmt='mp3'):
    lead = np.zeros((int(sr * SPRITE_LEAD), 2))
    gap = np.zeros((int(sr * SPRITE_GAP), 2))
    parts = []
    offsets = []
    pos = 0

    for seg in segments:
        if seg.ndim == 1:
            seg = to_stereo(seg, 0)
        block = np.concatenate([lead, seg])
        offsets.append([round(pos / sr, 6), round(len(block) / sr, 6)])
        parts.append(block)
        parts.append(gap)
        pos += len(block) + len(gap)

    audio = np.concatenate(parts) if parts else np.zeros((1, 2))
    export_audio(audio, os.path.join(out_dir, f'{inst_id}.{fmt}'), sr)

    return {
        'sprite': f'{inst_id}.{fmt}',
        'spriteDuration': round(len(audio) / sr, 6),
        'offsets': offsets,
    }


def write_manifest(out_dir, config, instruments, fmt='mp3'):
    manifest = dict(config)
    manifest['format'] = fmt
    manifest['sampleRate'] = SR
    manifest['instruments'] = instruments
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)


FORMANTS = {
    'a': [(730, 90), (1090, 110), (2440, 170), (3400, 250)],
    'e': [(530, 80), (1840, 150), (2480, 180), (3320, 250)],
    'i': [(270, 60), (2290, 200), (3010, 200), (3300, 250)],
    'o': [(570, 80), (840, 100), (2410, 170), (3400, 250)],
    'u': [(300, 60), (870, 100), (2240, 170), (3200, 250)],
    'schwa': [(500, 100), (1500, 130), (2500, 200), (3400, 250)],
    'm': [(280, 90), (1100, 180), (2200, 250), (3200, 300)],
    'n': [(280, 90), (1700, 200), (2600, 250), (3300, 300)],
}


def fold_to_voice(freq, lo=75, hi=880):
    f = float(freq)
    while f < lo:
        f *= 2
    while f > hi:
        f /= 2
    return f


def scale_tract(formants, factor):
    return [(f * factor, bw * max(0.6, factor)) for f, bw in formants]


def smooth_noise(n, cutoff, sr=SR):
    x = lowpass(np.random.randn(n), cutoff, sr)
    spread = np.std(x)
    return x / spread if spread > 0 else x


def f0_contour(freq, dur, sr=SR, vibrato_rate=5.0, vibrato_cents=0,
               vibrato_onset=0.35, jitter=0.005, drift_cents=0):
    n = int(sr * dur)
    t = np.arange(n) / sr
    semitones = np.zeros(n)

    if vibrato_cents:
        onset = np.clip(t / max(vibrato_onset, 1e-6), 0, 1)
        semitones += vibrato_cents / 100 * np.sin(2 * np.pi * vibrato_rate * t) * onset

    if drift_cents:
        semitones += drift_cents / 100 * np.linspace(0, 1, n)

    contour = freq * np.power(2, semitones / 12)
    if jitter:
        contour = contour * (1 + smooth_noise(n, 25, sr) * jitter)
    return contour


def glottal_flow(contour, sr=SR, open_quotient=0.6, shimmer=0.03, aspiration=0.0):
    n = len(contour)
    cycle = np.cumsum(contour / sr) % 1.0
    oq = open_quotient
    opening = oq * 0.8

    flow = np.zeros(n)
    rising = cycle < opening
    closing = (cycle >= opening) & (cycle < oq)
    x = np.clip(cycle / opening, 0, 1)
    flow[rising] = (3 * x ** 2 - 2 * x ** 3)[rising]
    y = np.clip((cycle - opening) / (oq - opening), 0, 1)
    flow[closing] = (1 - y ** 2)[closing]

    if shimmer:
        flow = flow * (1 + smooth_noise(n, 12, sr) * shimmer)

    source = np.diff(flow, prepend=0)
    if aspiration:
        breath = highpass(np.random.uniform(-1, 1, n), 300, sr)
        source = source + breath * aspiration * np.sqrt(np.mean(source ** 2)) * 4
    return source


def vocal_tract(source, start, end=None, sr=SR, block=256):
    if end is None:
        end = start
    out = np.asarray(source, dtype=float)
    n = len(out)
    nblocks = max(1, int(np.ceil(n / block)))

    for fi in range(min(len(start), len(end))):
        f_a, bw_a = start[fi]
        f_b, bw_b = end[fi]
        filtered = np.zeros(n)
        zi = np.zeros(2)
        for b in range(nblocks):
            lo = b * block
            hi = min(n, lo + block)
            if lo >= hi:
                break
            k = b / max(1, nblocks - 1)
            f = float(np.clip(f_a + (f_b - f_a) * k, 30, sr / 2 - 200))
            bw = float(np.clip(bw_a + (bw_b - bw_a) * k, 20, 1000))
            r = np.exp(-np.pi * bw / sr)
            a1 = 2 * r * np.cos(2 * np.pi * f / sr)
            a2 = -r * r
            b0 = 1 - a1 - a2
            filtered[lo:hi], zi = lfilter([b0], [1, -a1, -a2], out[lo:hi], zi=zi)
        out = filtered

    return out


def generate_reverb_ir(length_s, dark=0.8, sr=SR):
    n = int(sr * length_s)
    ir = np.zeros((n, 2))
    for ch in range(2):
        d = np.random.randn(n) * np.power(np.linspace(1, 0, n), 2.5)
        for i in range(1, n):
            d[i] = d[i] * (1 - dark) + d[i - 1] * dark
        ir[:, ch] = d
    return normalize(ir, 0.85)
