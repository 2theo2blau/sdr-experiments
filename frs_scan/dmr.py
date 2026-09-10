from fractions import Fraction

import numpy as np
from scipy import signal

from frs_scan.nfm import burst_body

BAUD = 4800.0

# One TDMA slot: 288 bits at 2 bits/symbol, 30 ms long
# CACH(12) + info(54) + sync or EMB(24) + info(54) dibits
BURST_SYMBOLS = 144
SYNC_SYMBOLS = 24
SYNC_OFFSET = 66 # dibits from the start of a burst to the sync field
SLOT_TYPE_SYMBOLS = 5 # slot type is 10 bits either side of the sync field

DEV_OUTER = 1944.0 # deviation of symbols +3 and -3
DEV_INNER = 648.0 # and of +1 and -1

# Symbol to dibit, in deviation order from +3 down to -3
DIBITS = np.array([0b01, 0b00, 0b10, 0b11], dtype=np.uint8)

# 48-bit sync patterns
SYNCS = {
    "755FD7DF75F7": "bs voice",
    "DFF57D75DF5D": "bs data",
    "7F7D5DD57DFD": "ms voice",
    "D5D7F77FD757": "ms data",
    "77D55F7DFD77": "ms rc",
    "5D577F7757FF": "direct slot1 voice",
    "F7FDD5DDFD55": "direct slot1 data",
    "7DFFD5F55D5F": "direct slot2 voice",
    "D7557F5FF7F5": "direct slot2 data",
}

# Data types carried in the slot type field of a data burst
DATA_TYPES = {
    0: "pi header", 1: "voice lc header", 2: "terminator with lc",
    3: "csbk", 4: "mbc header", 5: "mbc continuation",
    6: "data header", 7: "rate 1/2 data", 8: "rate 3/4 data",
    9: "idle", 10: "rate 1 data", 11: "usb data",
}


# DEMODULATION

def discriminator(y, fs):
    """
    Channelized baseband -> instantaneous frequency in Hz.

    Nothing is filtered after the phase difference.
    """
    d = np.angle(y[1:] * np.conj(y[:-1])).astype(np.float32)
    return d * (fs / (2 * np.pi))


def matched_filter(f, fs, baud=BAUD, cutoff_frac=1.15):
    """
    Band limit the discriminator output to the symbol rate.

    Cutoff is above Nyquist for the symbol rate rather than at it: DMR shapes
    its symbols at the transmitter, so this only has to remove what is above
    the signal.
    """
    taps = signal.firwin(65, cutoff_frac * (baud / 2) / (fs / 2))
    return signal.lfilter(taps, 1.0, f)[taps.size // 2:].astype(np.float32)


def _interp(x, t):
    """Linearly interpolate x at fractional index t."""
    i = int(t)
    a = t - i
    return x[i] * (1.0 - a) + x[i + 1] * a


def symbol_clock(x, sps=2.0, kp=0.02, ki=1e-4, max_drift=0.01):
    """
    Gardner timing recovery on a real signal at `sps` samples per symbol.

    The error term needs a sample halfway between two symbol instants, so it
    wants exactly 2 sps and a PI loop that corrects both the phase of the
    clock and its rate
    """
    if x.size < 4 * sps + 4:
        return np.zeros(0, dtype=np.float32), np.zeros(0)
    step = float(sps)
    nominal = float(sps)
    t = 2.0 * sps
    out, times = [], []
    prev = _interp(x, t - step)
    scale = float(np.percentile(np.abs(x), 90)) + 1e-9
    while t + step + 2.0 < x.size:
        cur = _interp(x, t)
        mid = _interp(x, t - 0.5 * step)
        out.append(cur)
        times.append(t)
        # the midsample is zero when the clock is on time
        e = float(mid * (prev - cur)) / (scale * scale)
        e = max(-1.0, min(1.0, e))
        step += ki * e # integral: tracks clock rate
        step = min(max(step, nominal * (1 - max_drift)), nominal * (1 + max_drift))
        t += step + kp * e # proportional: tracks clock phase
        prev = cur
    return np.asarray(out, dtype=np.float32), np.asarray(times)


def symbol_stream(y, fs, baud=BAUD):
    """
    Channelized baseband -> one sample per symbol, in Hz of deviation.

    Resamples to 2 samples/symbol because that is what Gardner needs, then
    lets the loop pick the symbol instants.
    """
    if y.size < 4 * int(fs / baud) + 128: # shorter than the filters
        return np.zeros(0, dtype=np.float32)
    f = matched_filter(discriminator(y, fs), fs, baud)
    ratio = Fraction(2 * baud / fs).limit_denominator(2000)
    f = signal.resample_poly(f, ratio.numerator, ratio.denominator)
    s, _ = symbol_clock(f)
    return s


def levels(s):
    """
    Center and scale a symbol stream so the outer levels sit at +/-3.

    The center is the receiver's frequency error (midpoint of the extremes).
    The payload's own DC balance would pull a mean away from it. Scale comes from 
    the outer levels.
    """
    if s.size == 0:
        return s.astype(np.float32)
    lo, hi = np.percentile(s, (2.0, 98.0))
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return ((s - mid) / (half + 1e-20) * 3.0).astype(np.float32)


def slice_dibits(x):
    """Four-level slicer: normalized symbols -> dibits."""
    idx = 3 - np.digitize(x, [-2.0, 0.0, 2.0])
    return DIBITS[idx]


def eye_quality(x):
    """
    Mean distance from a symbol to its level, in units where the levels are 2
    apart. Below about 0.25 the dibits are worth trusting.
    """
    if x.size == 0:
        return float("nan")
    nearest = np.array([-3.0, -1.0, 1.0, 3.0])[np.digitize(x, [-2.0, 0.0, 2.0])]
    return float(np.mean(np.abs(x - nearest)) / 2.0)


# FRAMING

def _hex_to_dibits(h):
    bits = bin(int(h, 16))[2:].zfill(len(h) * 4)
    return np.array([int(bits[i:i + 2], 2) for i in range(0, len(bits), 2)],
                    dtype=np.uint8)


SYNC_DIBITS = {name: _hex_to_dibits(h) for h, name in SYNCS.items()}


def find_syncs(dibits, max_errors=2):
    """
    Every sync pattern in the stream, as (index, name).

    A couple of wrong dibits are tolerated because a sync that only matches
    when perfect is lost at the edge of the signal, and 24 dibits
    is long enough that two errors do not make a false positive.
    """
    if dibits.size < SYNC_SYMBOLS:
        return []
    w = np.lib.stride_tricks.sliding_window_view(dibits, SYNC_SYMBOLS)
    hits = []
    for name, pat in SYNC_DIBITS.items():
        for i in np.flatnonzero((w != pat).sum(axis=1) <= max_errors):
            hits.append((int(i), name))
    return sorted(hits)


def _dibits_to_int(d):
    v = 0
    for x in d:
        v = (v << 2) | int(x)
    return v


def slot_type(dibits, sync_at):
    """
    The color code and data type on either side of a sync, as a dict, or None.

    These are the 10 bits immediately before the sync field and the 10
    immediately after it: 4 bits of color code, 4 of data type, and 12 of
    Golay parity that this reads past.
    """
    a, b = sync_at - SLOT_TYPE_SYMBOLS, sync_at + SYNC_SYMBOLS
    if a < 0 or b + SLOT_TYPE_SYMBOLS > dibits.size:
        return None
    v = _dibits_to_int(np.concatenate((dibits[a:sync_at],
                                       dibits[b:b + SLOT_TYPE_SYMBOLS])))
    dt = (v >> 12) & 0xF
    return {"color_code": (v >> 16) & 0xF,
            "data_type": dt,
            "data_type_name": DATA_TYPES.get(dt, f"reserved {dt}")}


def frame_bursts(dibits, syncs=None):
    """
    Cut the stream into 144-symbol bursts on the grid the syncs imply.

    Every sync in a transmission lands on the same grid, so the offset they
    agree on is the burst boundary, and bursts with no sync of their own still 
    land on it.
    """
    syncs = find_syncs(dibits) if syncs is None else syncs
    if not syncs:
        return []
    starts = [(i - SYNC_OFFSET) % BURST_SYMBOLS for i, _ in syncs]
    offset = int(np.bincount(starts, minlength=BURST_SYMBOLS).argmax())
    named = dict(syncs)

    out = []
    for a in range(offset, dibits.size - BURST_SYMBOLS + 1, BURST_SYMBOLS):
        sync_at = a + SYNC_OFFSET
        name = named.get(sync_at)
        burst = {"start": a, "sync": name,
                 "dibits": dibits[a:a + BURST_SYMBOLS]}
        if name is not None and name.endswith("data"):
            burst.update(slot_type(dibits, sync_at) or {})
        out.append(burst)
    return out


def demod_dmr(y, fs, baud=BAUD, trim=True):
    """
    Channelized baseband -> symbols, dibits and bursts.

    `y` must already be mixed to DC and band-limited to the channel.
    """
    if trim:
        y = y[burst_body(np.abs(y), fs)]
    s = symbol_stream(y, fs, baud)
    x = levels(s)
    dibits = slice_dibits(x)
    syncs = find_syncs(dibits)
    # A spectrum the wrong way round mirrors every level; the syncs say so.
    inverted = False
    if not syncs:
        x = -x
        dibits = slice_dibits(x)
        syncs = find_syncs(dibits)
        inverted = bool(syncs)
    return {"deviation": s, "symbols": x, "dibits": dibits,
            "syncs": syncs, "bursts": frame_bursts(dibits, syncs),
            "eye": eye_quality(x), "inverted": inverted}
