from fractions import Fraction

import numpy as np
from scipy import signal

# working rate: wide enough for a 12.5 kHz NFM channel
CHAN_RATE = 50000.0

# Half-bandwidth actually used, so the decimation filters need not protect anything wider.
KEEP_HZ = 8000.0

AUDIO_RATE = 24000 # demodulated audio
DEVIATION = 2500.0 # nominal peak deviation
DEEMPH = 750e-6 # land mobile is 750 us


def longest_run(mask):
    """Slice covering the longest contiguous True run, or None."""
    if not mask.any():
        return None
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    starts, stops = edges[0::2], edges[1::2]
    i = np.argmax(stops - starts)
    return slice(int(starts[i]), int(stops[i]))


def burst_body(env, fs, smooth_s=0.01, min_frac=0.1):
    """Slice covering the stretch where the carrier is actually up."""
    n = env.size
    k = max(1, int(fs * smooth_s))
    if n < 4 * k:
        return slice(0, n)
    s = np.convolve(env, np.ones(k, dtype=np.float32) / k, mode="same")
    lo, hi = np.percentile(s[k:n - k], (5, 95))
    if hi < 2 * lo: # no step: nothing to trim to
        return slice(0, n)
    seg = longest_run(s > np.sqrt(lo * hi))
    if seg is None:
        return slice(0, n)
    a, b = seg.start + k, seg.stop - k # keep clear of the ramps
    if b - a < max(512, int(min_frac * n)):
        return slice(0, n)
    return slice(a, b)

def demod_nfm(y, fs, audio_rate=AUDIO_RATE, deviation=DEVIATION, deemph=DEEMPH):
    """
    Channelized baseband -> audio.

    `y` must already be mixed to DC and band-limited.
    """
    d = np.angle(y[1:] * np.conj(y[:-1])).astype(np.float32)
    d *= fs / (2 * np.pi * deviation)

    # de-emphasis
    a = np.exp(-1.0 / (fs * deemph))
    d = signal.lfilter([1 - a], [1, -a], d)

    # audio band limit and resample
    d = signal.lfilter(signal.firwin(129, 3400.0 / (fs / 2)), 1.0, d)
    frac2 = Fraction(audio_rate / fs).limit_denominator(2000)
    d = signal.resample_poly(d, frac2.numerator, frac2.denominator)
    return d.astype(np.float32)
