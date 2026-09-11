import argparse
import os
import queue
import sys
import threading
import time
from datetime import datetime
from typing import Any, BinaryIO

import numpy as np

from frs_scan.classify import (add_classifier_args, carrier_stats, classify,
                               cut_kwargs, features, keep_set, prescreen,
                               ALL_KINDS)
from frs_scan.dsp import extract_channel_raw
from frs_scan.iq import RAW, write_cs8, write_wav
from frs_scan.nfm import AUDIO_RATE, CHAN_RATE, KEEP_HZ, demod_nfm


def parse_ranges(spec: str | None) -> list[tuple[float, float]]:
    """'470-480,462.36' -> [(lo, hi), ...] in Hz. Bare values get +/-10 kHz."""
    out = []
    for tok in filter(None, (spec or "").split(",")):
        if "-" in tok.lstrip("-"):
            a, b = tok.split("-", 1)
            lo, hi = float(a), float(b)
        else:
            lo = hi = float(tok)
        lo = lo * 1e6 if lo < 1e6 else lo
        hi = hi * 1e6 if hi < 1e6 else hi
        if lo == hi:
            lo, hi = lo - 10e3, hi + 10e3
        out.append((lo, hi))
    return out


class Ring:
    """Fixed-size ring of raw interleaved bytes, addressed by sample index."""

    def __init__(self, capacity_samples: float) -> None:
        self.cap = int(capacity_samples)
        self.buf = np.zeros(self.cap * 2, dtype=RAW)
        self.total = 0 # samples ever written

    def push(self, raw: np.ndarray) -> None:
        n = raw.size // 2
        if n >= self.cap:
            self.buf[:] = raw[-self.cap * 2:]
            self.total += n
            return
        p = (self.total % self.cap) * 2
        end = p + n * 2
        if end <= self.buf.size:
            self.buf[p:end] = raw
        else:
            k = self.buf.size - p
            self.buf[p:] = raw[:k]
            self.buf[: end - self.buf.size] = raw[k:]
        self.total += n

    def runs(self, start: int, stop: int) -> list[np.ndarray] | None:
        """Samples [start, stop) as one or two contiguous views, or None."""
        oldest = max(0, self.total - self.cap)
        if start < oldest or stop > self.total or stop <= start:
            return None
        p = (start % self.cap) * 2
        end = p + (stop - start) * 2
        if end <= self.buf.size:
            return [self.buf[p:end]]
        return [self.buf[p:], self.buf[:end - self.buf.size]]

    def get(self, start: int, stop: int) -> np.ndarray | None:
        """Samples [start, stop) copied out by absolute index, or None."""
        parts = self.runs(start, stop)
        return None if parts is None else np.concatenate(parts)


def channelize(parts: list[np.ndarray], fs: float, offset_hz: float, out_rate: float, keep_hz: float = KEEP_HZ) -> tuple[np.ndarray, float]:
    """extract_channel_raw across the ring's one or two contiguous runs."""
    ys, n0 = [], 0
    for part in parts:
        y, cfs = extract_channel_raw(part, fs, offset_hz, out_rate, keep_hz,
                                     n_start=n0)
        ys.append(y)
        n0 += part.size // 2
    return (ys[0] if len(ys) == 1 else np.concatenate(ys)), cfs


def reader(stdin: BinaryIO, ring: Ring, nbytes: int, frames: queue.Queue, stats: dict[str, Any]) -> None:
    """
    Drain stdin into the ring forever, whatever else is going on.

    This thread must never block for long, hackrf_transfer dies if the pipe
    goes undrained for 1 second.
    """
    try:
        import fcntl
        cap = int(open("/proc/sys/fs/pipe-max-size").read())
        fcntl.fcntl(stdin.fileno(), 1031, cap) # F_SETPIPE_SZ
    except Exception:
        pass
    fd = stdin.fileno()
    while True:
        chunks, got = [], 0
        while got < nbytes:
            part = os.read(fd, nbytes - got)
            if not part:
                break
            chunks.append(part)
            got += len(part)
        buf = b"".join(chunks)
        if got < nbytes:
            break
        # longest stretch spent *outside* read()
        t_out = time.perf_counter()
        base = ring.total
        ring.push(np.frombuffer(buf, dtype=RAW))
        try:
            frames.put_nowait(base)
        except queue.Full:
            stats["dropped"] += 1
        stats["worst_gap"] = max(stats["worst_gap"], time.perf_counter() - t_out)
    frames.put(None)


def group_bins(mask: np.ndarray, min_bins: int, gap: int) -> list[tuple[int, int]]:
    """Contiguous runs of True, merging gaps up to `gap` bins."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > gap + 1)
    groups, start = [], 0
    for s in list(splits) + [idx.size - 1]:
        a, b = idx[start], idx[s]
        if (b - a + 1) >= min_bins:
            groups.append((int(a), int(b)))
        start = s + 1
    return groups


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-f", "--center", type=float, required=True, help="Hz")
    ap.add_argument("-s", "--rate", type=float, required=True, help="Hz")
    ap.add_argument("--bin-hz", type=float, default=5000.0, help="FFT resolution")
    ap.add_argument("--frame-ms", type=float, default=50.0, help="detection interval")
    ap.add_argument("--avg", type=int, default=8, help="FFTs averaged per frame")
    ap.add_argument("-t", "--threshold", type=float, default=10.0, help="dB over floor")
    ap.add_argument("--min-bins", type=int, default=2, help="reject narrower spurs")
    ap.add_argument("--min-len", type=float, default=0.3, help="shortest burst, s")
    ap.add_argument("--max-gap", type=float, default=0.5, help="merge gaps, s")
    ap.add_argument("--exclude", help="e.g. '470-480,462.36'")
    ap.add_argument("--save-dir", help="write channelized IQ of kept bursts here")
    ap.add_argument("--wav-dir", help="write WAVs of kept bursts here (default: --save-dir)")
    ap.add_argument("--pre", type=float, default=0.4, help="pre-roll, s")
    ap.add_argument("--post", type=float, default=0.4, help="post-roll, s")
    ap.add_argument("--max-save", type=float, default=12.0,
                    help="cap snippet length, s (drives RAM use)")
    ap.add_argument("--csv", default="bursts.csv")
    ap.add_argument("--settle", type=float, default=3.0, help="floor learning time, s")
    ap.add_argument("--keep", default="voice",
                    metavar="KINDS",
                    help="kinds to write: voice, morse, signal, all, or a "
                         "comma-separated list (default: voice, which is "
                         "voice+morse)")
    add_classifier_args(ap)
    ap.add_argument("--channel-rate", type=float, default=CHAN_RATE,
                    help="saved IQ sample rate after mixing to the burst, Hz")
    ap.add_argument("--classify-s", type=float, default=0.5,
                    help="seconds from the burst body used for the cheap screen")
    ap.add_argument("--classify-max", type=float, default=2.0,
                    help="seconds of a burst used to decide what it is")
    args = ap.parse_args()
    want = keep_set(args.keep)
    cuts = cut_kwargs(args)
    wav_dir = args.wav_dir or args.save_dir

    fs, fc = args.rate, args.center
    nfft = 1 << int(np.ceil(np.log2(fs / args.bin_hz)))
    block = max(nfft * args.avg, int(fs * args.frame_ms / 1000.0))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs)) + fc
    win = np.hanning(nfft).astype(np.float32)
    dt = block / fs

    bin_keep = np.ones(nfft, dtype=bool)
    for lo, hi in parse_ranges(args.exclude):
        bin_keep &= ~((freqs >= lo) & (freqs <= hi))

    maxframes = max(16, int(4.0 / dt))
    queue_s = maxframes * dt

    ring_s = args.pre + args.max_save + args.post + queue_s + 5
    ring = Ring(fs * ring_s)
    sys.stderr.write(f"# ring buffer {ring_s:.1f}s = {ring.buf.nbytes/1e6:.0f} MB "
                     f"({args.pre + args.max_save + args.post:.1f}s snippet + "
                     f"{queue_s:.1f}s queue + 5.0s slack)\n")
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    if wav_dir:
        os.makedirs(wav_dir, exist_ok=True)

    csv = open(args.csv, "a", buffering=1)
    if csv.tell() == 0:
        csv.write("iso_time,elapsed_s,freq_hz,duration_s,peak_snr_db,kind,file\n")

    sys.stderr.write(
        f"# {fs/1e6:.1f} Msps, {nfft}-pt FFT, {fs/nfft/1e3:.2f} kHz bins, "
        f"{dt*1000:.0f} ms frames, {(freqs[0])/1e6:.3f}-{(freqs[-1])/1e6:.3f} MHz\n"
        f"# learning noise floor for {args.settle:.0f} s...\n")

    nbytes = block * 2
    offs = np.linspace(0, block - nfft, args.avg).astype(int)
    floor = None
    active = {} # key -> dict(start_idx, last_idx, peak, fsum, wsum)
    t0 = time.time()
    t_read = t_work = 0.0 # wall time blocked on stdin vs. spent processing
    nframes = 0
    nkind = {k: 0 for k in ALL_KINDS}
    rstats = {"dropped": 0, "worst_gap": 0.0}
    frames = queue.Queue(maxsize=maxframes)
    rd = threading.Thread(target=reader, args=(sys.stdin.buffer, ring, nbytes, frames, rstats), daemon=True)
    rd.start()

    try:
        while True:
            tick = time.perf_counter()
            frame = frames.get()
            t_read += time.perf_counter() - tick
            if frame is None:
                break
            tick = time.perf_counter()
            raw = ring.get(frame, frame + block)
            if raw is None: # lapped while busy, skip it
                rstats["dropped"] += 1
                continue
            frame_end = frame + block
            nframes += 1

            acc = np.zeros(nfft, dtype=np.float64)
            for o in offs:
                seg = raw[o * 2: (o + nfft) * 2].astype(np.float32)
                x = (seg[0::2] + 1j * seg[1::2]) * win
                acc += np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
            psd = 10 * np.log10(acc / len(offs) + 1e-12)

            if floor is None:
                floor = psd.copy()
                continue
            settling = (nframes * dt) < args.settle
            if settling:
                floor = 0.9 * floor + 0.1 * psd
                continue

            # per-bin floor
            hot = psd > floor + args.threshold
            floor = np.where(hot, floor + 0.001 * (psd - floor),
                             floor + 0.02 * (psd - floor))

            snr = psd - floor
            mask = hot & bin_keep
            seen = set()
            for a, b in group_bins(mask, args.min_bins, 1):
                w = np.maximum(snr[a:b + 1], 0.0)
                cf = float((freqs[a:b + 1] * w).sum() / (w.sum() + 1e-12))
                key = round(cf / 12500.0)
                seen.add(key)
                ev = active.get(key)
                if ev is None:
                    active[key] = dict(start=frame, last=frame_end,
                                       peak=float(snr[a:b + 1].max()),
                                       fsum=cf * w.sum(), wsum=w.sum())
                else:
                    ev["last"] = frame_end
                    ev["peak"] = max(ev["peak"], float(snr[a:b + 1].max()))
                    ev["fsum"] += cf * w.sum()
                    ev["wsum"] += w.sum()

            for key in [k for k in active if k not in seen]:
                ev = active[key]
                if (frame_end - ev["last"]) / fs < args.max_gap:
                    continue
                del active[key]
                dur = (ev["last"] - ev["start"]) / fs
                if dur < args.min_len:
                    continue
                cf = ev["fsum"] / (ev["wsum"] + 1e-12)
                elapsed = ev["start"] / fs

                a = max(0, int(ev["start"] - args.pre * fs))
                b = min(frame_end,
                        int(ev["last"] + args.post * fs),
                        int(ev["start"] + args.max_save * fs))
                pa = min(int(ev["start"] + 0.1 * fs),
                         max(a, b - int(args.classify_s * fs)))
                probe = ring.runs(max(a, pa),
                                  min(b, pa + int(args.classify_s * fs)))
                if probe is None: # aged out of the ring
                    continue

                # screen on a short window from the burst body
                y, cfs = channelize(probe, fs, cf - fc, args.channel_rate)
                feats = carrier_stats(y, cfs)
                kind = prescreen(feats, args.cv_cut)

                audio = None
                if kind is None or kind in want:
                    # classify on a bounded window
                    cw = ring.runs(a, min(b, a + int(args.classify_max * fs)))
                    if cw is None:
                        continue
                    y, cfs = channelize(cw, fs, cf - fc, args.channel_rate)
                    audio = demod_nfm(y, cfs)
                    feats = features(y, cfs, audio)
                    kind = classify(feats, **cuts)
                nkind[kind] += 1

                path = ""
                if audio is not None and kind in want:
                    # worth keeping
                    full = ring.runs(a, b)
                    if full is not None and b - a > int(args.classify_max * fs):
                        y, cfs = channelize(full, fs, cf - fc, args.channel_rate)
                        audio = demod_nfm(y, cfs)
                    # the reader kept writing while that ran 
                    if full is None or ring.total - a > ring.cap:
                        rstats["dropped"] += 1
                        continue
                    stem = (f"{datetime.now():%Y%m%d_%H%M%S}_"
                            f"{cf/1e6:.4f}MHz_{dur:.1f}s_{kind}")
                    if args.save_dir:
                        path = os.path.join(args.save_dir, stem + ".iq")
                        write_cs8(path, y)
                    if wav_dir:
                        write_wav(os.path.join(wav_dir, stem + ".wav"),
                                  audio, AUDIO_RATE)
                stamp = datetime.now().isoformat(timespec="seconds")
                print(f"{stamp}  {cf/1e6:10.4f} MHz  {dur:5.1f}s  "
                      f"{ev['peak']:5.1f} dB  cv {feats['env_cv']:.2f}  "
                      f"kurt {feats['kurt']:6.2f}  "
                      f"voiced {feats.get('voiced', float('nan')):.2f}  "
                      f"tone {feats.get('tone', float('nan')):.2f}"
                      f"  {kind:8} {path}", flush=True)
                csv.write(f"{stamp},{elapsed:.2f},{cf:.0f},{dur:.2f},"
                          f"{ev['peak']:.1f},{kind},{path}\n")

            t_work += time.perf_counter() - tick

            if nframes % int(30 / dt) == 0:
                wall = time.time() - t0
                streamed = ring.total / fs
                sys.stderr.write(
                    f"# {streamed:7.0f}s streamed, {wall:7.0f}s wall, "
                    f"x{streamed/wall:.2f}, cpu {t_work/wall:.2f}, "
                    f"idle {t_read/wall:.2f}, drop {rstats['dropped']}, "
                    f"gap {rstats['worst_gap']*1000:.0f} ms   "
                    + " ".join(f"{k} {v}" for k, v in nkind.items()) + "\n")
    except KeyboardInterrupt:
        pass
    finally:
        csv.close()
        wall = max(time.time() - t0, 1e-9)
        sys.stderr.write(
            f"# stopped after {ring.total/fs:.0f}s of stream, cpu {t_work/wall:.2f}, "
            f"drop {rstats['dropped']}, worst read gap "
            f"{rstats['worst_gap']*1000:.0f} ms, "
            + " ".join(f"{k} {v}" for k, v in nkind.items()) + "\n")


if __name__ == "__main__":
    main()
