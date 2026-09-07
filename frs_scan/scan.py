import argparse
import os
import sys

import numpy as np

from frs_scan.classify import (add_classifier_args, classify, cut_kwargs,
                               features, keep_set)
from frs_scan.iq import FORMATS, file_length, load_iq, write_wav
from frs_scan.dsp import extract_channel
from frs_scan.nfm import AUDIO_RATE, CHAN_RATE, KEEP_HZ, demod_nfm

# FRS/GMRS channel plan (MHz). 1-7 shared, 8-14 FRS low power, 15-22 GMRS main
FRS_GMRS = {
    1: 462.5625, 2: 462.5875, 3: 462.6125, 4: 462.6375,
    5: 462.6625, 6: 462.6875, 7: 462.7125,
    8: 467.5625, 9: 467.5875, 10: 467.6125, 11: 467.6375,
    12: 467.6625, 13: 467.6875, 14: 467.7125,
    15: 462.5500, 16: 462.5750, 17: 462.6000, 18: 462.6250,
    19: 462.6500, 20: 462.6750, 21: 462.7000, 22: 462.7250,
}


def detect(path, fmt, fs, fc, channels, nfft=4096, hop_blocks=64):
    """Pass 1: per-channel power over time, from averaged spectrogram blocks.

    Returns (times, {name: power_db_array}, noise_floor_db).
    """
    total = file_length(path, fmt)
    block = nfft * hop_blocks
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs)) + fc
    win = np.hanning(nfft).astype(np.float32)

    # map each channel to the fft bins covering its 12.5 kHz slot.
    bins = {}
    for name, f0 in channels.items():
        sel = np.where(np.abs(freqs - f0) <= 6250)[0]
        if sel.size:
            bins[name] = sel

    times, power = [], {k: [] for k in bins}
    pos = 0
    while pos + block <= total:
        x = load_iq(path, fmt, pos, block)
        if x.size < block:
            break
        frames = x[: (x.size // nfft) * nfft].reshape(-1, nfft) * win
        psd = np.abs(np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)) ** 2
        psd = psd.mean(axis=0)
        for name, sel in bins.items():
            power[name].append(psd[sel].sum())
        times.append(pos / fs)
        pos += block

    power = {k: 10 * np.log10(np.array(v) + 1e-20) for k, v in power.items()}
    allp = np.concatenate([v for v in power.values()]) if power else np.array([0.0])
    return np.array(times), power, float(np.median(allp))


def find_bursts(times, power_db, floor_db, thresh_db, min_len, max_gap):
    """Group consecutive above-threshold frames into bursts."""
    if times.size < 2:
        return []
    dt = times[1] - times[0]
    hot = power_db > (floor_db + thresh_db)
    bursts, start, last = [], None, None
    for i, h in enumerate(hot):
        if h:
            if start is None:
                start = i
            last = i
        elif start is not None and (i - last) * dt > max_gap:
            if (times[last] - times[start]) >= min_len:
                bursts.append((times[start], times[last] + dt,
                               float(power_db[start:last + 1].max())))
            start = None
    if start is not None and (times[last] - times[start]) >= min_len:
        bursts.append((times[start], times[last] + dt,
                       float(power_db[start:last + 1].max())))
    return bursts


def dedupe(cands, channels, guard=30000.0):
    """
    Drop bleed into neighbouring channel bins

    Two hits overlapping in time and within `guard` Hz -> keep the stronger
    """
    out = []
    for c in sorted(cands, key=lambda r: -r[3]):
        name, t0, t1, pk = c
        if any(abs(channels[name] - channels[o[0]]) <= guard
               and t0 < o[2] and o[1] < t1 for o in out):
            continue
        out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("-f", "--center", type=float, required=True, help="capture center freq, Hz")
    ap.add_argument("-s", "--rate", type=float, required=True, help="sample rate, Hz")
    ap.add_argument("--format", default="cs8", choices=list(FORMATS))
    ap.add_argument("--freqs", help="comma-separated extra freqs in Hz or MHz")
    ap.add_argument("--only", help="comma-separated channel names to check")
    ap.add_argument("-t", "--threshold", type=float, default=8.0, help="dB over noise floor")
    ap.add_argument("--min-len", type=float, default=0.4, help="shortest burst, seconds")
    ap.add_argument("--max-gap", type=float, default=0.6, help="merge gaps shorter than this")
    ap.add_argument("--pad", type=float, default=0.3, help="seconds of padding each side")
    ap.add_argument("-o", "--outdir", default="bursts")
    ap.add_argument("--keep", default="all", metavar="KINDS",
                    help="which classes to write WAVs for (default: all)")
    add_classifier_args(ap)
    args = ap.parse_args()
    keep = keep_set(args.keep)
    cuts = cut_kwargs(args)

    channels = {}
    for ch, mhz in FRS_GMRS.items():
        f0 = mhz * 1e6
        if abs(f0 - args.center) <= args.rate / 2 * 0.95:
            channels[f"ch{ch:02d}"] = f0
    if args.freqs:
        for i, tok in enumerate(args.freqs.split(",")):
            v = float(tok)
            v = v * 1e6 if v < 1e6 else v
            if abs(v - args.center) <= args.rate / 2 * 0.95:
                channels[f"f{i}_{v/1e6:.4f}"] = v
    if args.only:
        # which channels to look at; --keep still decides what gets written to audio dir
        want = set(args.only.split(","))
        channels = {k: v for k, v in channels.items() if k in want}

    if not channels:
        sys.exit("No channels fall inside the captured band. Check -f and -s.")

    total = file_length(args.capture, args.format)
    print(f"{args.capture}: {total/args.rate:.1f} s, "
          f"{os.path.getsize(args.capture)/1e9:.2f} GB, {len(channels)} channels\n")

    times, power, floor = detect(args.capture, args.format, args.rate,
                                 args.center, channels)
    print(f"noise floor {floor:.1f} dB, frame {times[1]-times[0]:.3f} s\n"
          if times.size > 1 else "")

    os.makedirs(args.outdir, exist_ok=True)
    
    cands = []
    for name in sorted(channels):
        for t0, t1, pk in find_bursts(times, power[name], floor,
                                      args.threshold, args.min_len, args.max_gap):
            cands.append((name, t0, t1, pk))
    cands = dedupe(cands, channels)

    rows, n = [], 0
    for name, t0, t1, pk in cands:
        a = max(0, int((t0 - args.pad) * args.rate))
        b = min(total, int((t1 + args.pad) * args.rate))
        x = load_iq(args.capture, args.format, a, b - a)
        if x.size < 1000:
            continue
        y, fs2 = extract_channel(x, args.rate, channels[name] - args.center,
                                 CHAN_RATE, KEEP_HZ)
        audio = demod_nfm(y, fs2)
        feats = features(y, fs2, audio)
        kind = classify(feats, **cuts)
        fn = ""
        if kind in keep:
            fn = os.path.join(args.outdir, f"{name}_{t0:07.1f}s_{kind}.wav")
            write_wav(fn, audio, AUDIO_RATE)
            n += 1
        rows.append((name, t0, t1 - t0, pk - floor, feats, kind, fn))

    if not rows:
        print("No bursts found. Lower -t, or check that the band was right.")
        return
    print(f"{'channel':10} {'start':>9} {'len':>6} {'SNR':>6} "
          f"{'env_cv':>7} {'kurt':>7} {'flat':>6} {'syl':>6}  verdict")
    print("-" * 78)
    for name, t0, dur, snr, f, kind, _ in sorted(rows, key=lambda r: r[1]):
        print(f"{name:10} {t0:8.1f}s {dur:5.1f}s {snr:5.1f}dB "
              f"{f['env_cv']:7.3f} {f['kurt']:7.2f} {f['flatness']:6.2f} "
              f"{f['syllabic']:6.2f}  {kind}")
    print(f"\n{len(rows)} bursts examined, {n} WAVs written to {args.outdir}/")


if __name__ == "__main__":
    main()
