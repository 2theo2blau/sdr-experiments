from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy import signal


def decim_stages(d, cap=25):
    """
    Split a decimation factor into stages of at most `cap`.
    """
    stages = []
    for p in (2, 3, 5, 7):
        while d % p == 0 and d > 1:
            if stages and stages[-1] * p <= cap:
                stages[-1] *= p
            else:
                stages.append(p)
            d //= p
    if d > 1:
        stages.append(d)
    return stages or [1]


def _stage_taps(d, fs_in, keep_hz, atten=60.0):
    """
    Anti-alias FIR for one decimation stage.

    A lowpass at half the new rate is overkill: only +/-keep_hz is ever looked at.

    Length is forced to a multiple of `d` plus one.
    """
    nyq = fs_in / 2.0
    wp = min(keep_hz / nyq, 0.4 / d)
    ws = max((fs_in / d - keep_hz) / nyq, wp * 1.5)
    n, beta = signal.kaiserord(atten, ws - wp)
    n = max(d, n - 1 + (-(n - 1)) % d) + 1
    return signal.firwin(n, (wp + ws) / 2, window=("kaiser", beta)).astype(np.float32)


def _polyphase(h, x, d, chunk=1 << 18):
    """
    Decimating FIR filter, exactly signal.upfirdn(h, x, 1, d), but faster (strictly on imaginary inputs).

    One matrix product per chunk, over a sliding-window view of the input taken
    every d-th row. Inner loop is handled by BLAS.

    Chunked because the (nout, len(h)) window view would be too big.
    """
    hr = h[::-1].astype(np.float32)
    nout = -(-(x.size + h.size - 1) // d)
    pad = (nout - 1) * d + 1 - x.size
    xp = np.concatenate((np.zeros(h.size - 1, x.dtype), x,
                         np.zeros(max(pad, 0), x.dtype)))
    out = np.empty(nout, dtype=x.dtype)
    for i in range(0, nout, chunk):
        m = min(chunk, nout - i)
        seg = xp[i * d: i * d + (m - 1) * d + h.size]
        out[i:i + m] = np.lib.stride_tricks.sliding_window_view(seg, h.size)[::d] @ hr
    return out


def _decimate(x, fs, stages, keep_hz):
    for d in stages:
        if d > 1:
            x = _polyphase(_stage_taps(d, fs, keep_hz), x, d)
            fs /= d
    return x


LO_TABLE = 1 << 16       # see _mix


def _mix(raw, fs, offset_hz, n0, table=LO_TABLE):
    """
    Interleaved int8 -> complex64, mixed down by offset_hz.

    Builds the complex array directly rather than via a float32 copy.

    The oscillator is one precomputed period tiled over the input. Tiling needs the mixing
    frequency to be an exact multiple of fs/table, so it is rounded to one.
    """
    x = np.empty(raw.size // 2, dtype=np.complex64)
    x.real = raw[0::2]
    x.imag = raw[1::2]
    if not offset_hz:
        return x
    period = _lo_period(round(-offset_hz / fs * table) % table, table)
    if period is not None:
        _apply_lo(x, period, n0)
    return x


@lru_cache(maxsize=64)
def _lo_period(k, table):
    """
    One period of exp(2j*pi*k*n/table), or None when k is 0.

    Cached: a channel is revisited many times in a capture, so we build this just once.
    """
    if k == 0:
        return None
    ph = (2 * np.pi / table) * ((k * np.arange(table)) % table)
    period = np.empty(table, dtype=np.complex64)
    np.cos(ph, out=period.real)
    np.sin(ph, out=period.imag)
    period.flags.writeable = False
    return period


def _apply_lo(x, period, n0):
    """
    Multiply x in place by the tiled oscillator, starting at absolute n0.
    """
    i, pos, table = 0, int(n0 % period.size), period.size
    while i < x.size:
        m = min(table - pos, x.size - i)
        x[i:i + m] *= period[pos:pos + m]
        i += m
        pos = (pos + m) % table
    return x


def _channel_filter(y, fs_out, keep_hz, ntaps=129):
    """
    Final channel filter at +/-keep_hz, on the already-decimated signal.

    The start-up transient is dropped, not returned. 
    """
    taps = signal.firwin(ntaps, keep_hz / (fs_out / 2))
    return signal.lfilter(taps, 1.0, y).astype(np.complex64)[taps.size:]


def extract_channel(x, fs, offset_hz, out_rate, keep_hz):
    """
    Mix `offset_hz` down to DC, band-limit, and decimate towards out_rate.

    Returns (y, fs_out). 
    """
    if offset_hz:
        # same rounded-frequency tiled oscillator as _mix
        period = _lo_period(round(-offset_hz / fs * LO_TABLE) % LO_TABLE, LO_TABLE)
        if period is not None:
            x = _apply_lo(x.astype(np.complex64, copy=True), period, 0)
    ratio = Fraction(out_rate / fs).limit_denominator(2000)
    if ratio.numerator == 1:
        y = _decimate(x, fs, decim_stages(ratio.denominator), keep_hz)
    else:
        y = signal.resample_poly(x, ratio.numerator, ratio.denominator)
    fs_out = fs * ratio.numerator / ratio.denominator
    return _channel_filter(y, fs_out, keep_hz), fs_out


def extract_channel_raw(raw, fs, offset_hz, out_rate, keep_hz,
                        n_start=0, block_samples=1 << 20):
    """
    extract_channel straight off interleaved int8, in bounded memory.

    Mixing and the first decimation stage run block by block, so we are not
    building a huge complex array of the whole wideband signal.
    """
    ratio = Fraction(out_rate / fs).limit_denominator(2000)
    fs_out = fs * ratio.numerator / ratio.denominator
    if ratio.numerator != 1:
        return extract_channel(_mix(raw, fs, offset_hz, n_start), fs,
                               0.0, out_rate, keep_hz)[0], fs_out

    stages = decim_stages(ratio.denominator)
    d1 = stages[0]
    h1 = _stage_taps(d1, fs, keep_hz)
    over = h1.size - 1 # a multiple of d1
    block = max(block_samples - block_samples % d1, d1)

    state = np.zeros(over, dtype=np.complex64)
    parts, total = [], raw.size // 2
    for n0 in range(0, total - total % d1, block):
        n = min(block, (total - total % d1) - n0)
        xin = np.concatenate((state, _mix(raw[n0 * 2:(n0 + n) * 2],
                                          fs, offset_hz, n_start + n0)))
        y = _polyphase(h1, xin, d1)
        parts.append(y[over // d1: over // d1 + n // d1])
        state = xin[-over:]
    y = np.concatenate(parts) if parts else np.zeros(0, np.complex64)
    return (_channel_filter(_decimate(y, fs / d1, stages[1:], keep_hz),
                            fs_out, keep_hz), fs_out)
