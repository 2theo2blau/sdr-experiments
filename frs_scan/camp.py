"""Camp one channel; write a continuous FM-discriminator WAV for dsd-fme."""
import argparse
import itertools
import os
import queue
import socket
import sys
import threading
import time
import wave
from fractions import Fraction

import numpy as np

from frs_scan.decode import DISC_RATE, scale_discriminator
from frs_scan.dmr import demod_dmr, discriminator
from frs_scan.dsp import extract_channel_raw
from frs_scan.iq import RAW
from frs_scan.nfm import KEEP_HZ

# Raw samples carried either side of a block. Everything in the chain is FIR,
# so this only has to outrun the longest impulse response in it
OVERLAP_S = 0.05

PROBE_S = 0.5 # slice of a block handed to the DMR health probe


def plan(fs, out_rate, block_s, overlap_s=OVERLAP_S):
    """(decimation, block, overlap) in samples. Multiples of D so seams land on sample boundaries."""
    ratio = Fraction(out_rate / fs).limit_denominator(2000)
    if ratio.numerator != 1:
        raise SystemExit(
            f"--rate {fs:.0f} does not decimate to {out_rate:.0f} by an integer.\n"
            f"Interpolating would drop extract_channel_raw onto its "
            f"resample_poly path, which builds the whole wideband block in "
            f"memory. Capture at a multiple of {out_rate:.0f}: "
            f"4.8e6 or 9.6e6 both work.")
    d = ratio.denominator
    if d == 1:
        raise SystemExit(
            "--rate equals --wav-rate; extract_channel_raw has no decimation "
            "stage to put its anti-alias filter in. Capture wider.")
    block = max(d, int(round(block_s * fs)) // d * d)
    overlap = max(d, -(-int(round(overlap_s * fs)) // d) * d)
    return d, block, overlap


def dc_estimate(raw):
    """Mean I/Q, rounded to ADC granularity."""
    return (int(round(float(raw[0::2].mean()))),
            int(round(float(raw[1::2].mean()))))


def remove_dc(raw, dc):
    """Subtract static DC from interleaved int8. Better not to camp on center."""
    if dc == (0, 0):
        return raw
    out = raw.astype(np.int16)
    out[0::2] -= dc[0]
    out[1::2] -= dc[1]
    return np.clip(out, -127, 127).astype(RAW)


def channel_stream(blocks, fs, offset_hz, out_rate, keep_hz, d, block, overlap,
                   dc_block=True):
    """
    Raw blocks -> (dev Hz, complex) chunks. Overlap-save. Emits a block once the 
    next one arrives.
    """
    m0 = overlap // d
    prev = np.zeros(overlap * 2, dtype=RAW)
    cur = None
    pos = 0 # absolute sample index of the start of `cur`
    for nxt in itertools.chain(blocks, [None]):
        if nxt is not None:
            keep = nxt.size // 2 // d * d # only whole output samples
            nxt = nxt[:keep * 2]
        if cur is None:
            cur = nxt
            continue
        if cur.size:
            tail = (nxt[:overlap * 2] if nxt is not None
                    else np.zeros(overlap * 2, dtype=RAW))
            win = np.concatenate((prev, cur, tail))
            dc = dc_estimate(cur) if dc_block else (0, 0)
            y, cfs = extract_channel_raw(remove_dc(win, dc), fs, offset_hz,
                                         out_rate, keep_hz,
                                         n_start=pos - overlap)
            n = cur.size // 2 // d
            f = discriminator(y, cfs)
            if m0 + n > f.size: # overlap too short for the filters
                raise RuntimeError("overlap does not cover the filter transient")
            yield f[m0:m0 + n], y[m0:m0 + n], cfs, front_end(cur) + dc
            prev = np.concatenate((prev, cur))[-overlap * 2:]
            pos += cur.size // 2
        cur = nxt
        if cur is None:
            break


def pcm16(x):
    """Scaled discriminator -> 16-bit samples."""
    return (np.clip(x, -1.0, 1.0) * 32767).astype("<i2")


def front_end(raw):
    """
    ADC health from the raw block: (rms, clip fraction). rms << 127 = too quiet; 
    clip > ~1% = saturating.
    """
    a = np.abs(raw.astype(np.int16))
    return (float(np.sqrt(np.mean(a.astype(np.float32) ** 2))),
            float(np.count_nonzero(a >= 127) / max(a.size, 1)))


class WavWriter:
    """16-bit mono WAV, no normalization."""

    def __init__(self, path, rate):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.w = wave.open(path, "wb")
        self.w.setnchannels(1)
        self.w.setsampwidth(2)
        self.w.setframerate(int(rate))
        self.frames = 0

    def write(self, pcm):
        self.w.writeframes(pcm.tobytes())
        self.frames += pcm.size

    def close(self):
        self.w.close()


class TcpSink:
    """TCP server for `dsd-fme -i tcp`. Raw le16 mono, no header. Slow clients get dropped."""

    def __init__(self, port, host="127.0.0.1", depth=32, stats=None):
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, port))
        self.srv.listen(4)
        self.depth = depth
        self.stats = stats if stats is not None else {}
        self.stats.setdefault("tcp_dropped", 0)
        self.clients = {} # queue -> socket
        self.lock = threading.Lock()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                sock, addr = self.srv.accept()
            except OSError:
                return
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            q = queue.Queue(maxsize=self.depth)
            with self.lock:
                self.clients[q] = sock
            threading.Thread(target=self._serve, args=(q, sock), daemon=True).start()
            sys.stderr.write(f"# tcp client connected from {addr[0]}:{addr[1]}\n")

    def _serve(self, q, sock):
        try:
            while True:
                buf = q.get()
                if buf is None:
                    break
                sock.sendall(buf)
        except OSError:
            pass
        finally:
            with self.lock:
                self.clients.pop(q, None)
            sock.close()
            sys.stderr.write("# tcp client gone\n")

    def write(self, pcm):
        buf = pcm.tobytes()
        with self.lock:
            qs = list(self.clients)
        for q in qs:
            try:
                q.put_nowait(buf)
            except queue.Full:
                self.stats["tcp_dropped"] += 1

    def count(self):
        with self.lock:
            return len(self.clients)

    def close(self):
        with self.lock:
            qs = list(self.clients)
        for q in qs:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        self.srv.close()


def reader(stdin, nbytes, blocks, stats):
    """stdin -> queue. Must not block for 1 second or hackrf_transfer dies."""
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
        t_out = time.perf_counter()
        try:
            blocks.put_nowait(b"".join(chunks))
        except queue.Full:
            stats["dropped"] += 1
        stats["worst_gap"] = max(stats["worst_gap"], time.perf_counter() - t_out)
        if got < nbytes:
            break
    blocks.put(None)


def stdin_blocks(nbytes, stats, maxsize=8):
    """Blocks read from a live pipe, on their own thread."""
    q = queue.Queue(maxsize=maxsize)
    t = threading.Thread(target=reader, args=(sys.stdin.buffer, nbytes, q, stats),
                         daemon=True)
    t.start()
    while True:
        buf = q.get()
        if buf is None:
            break
        yield np.frombuffer(buf, dtype=RAW)


def file_blocks(path, nbytes, pace_s=0.0):
    """Same blocks from a recorded capture."""
    with open(path, "rb") as fh:
        t = time.perf_counter()
        while True:
            buf = fh.read(nbytes)
            if not buf:
                break
            if pace_s:
                t += pace_s
                time.sleep(max(0.0, t - time.perf_counter()))
            yield np.frombuffer(buf, dtype=RAW)


def probe(y, fs, seconds=PROBE_S):
    """DMR health check. trim=False so quiet stretches don't look empty."""
    n = min(y.size, int(seconds * fs))
    if n < 1000:
        return None
    r = demod_dmr(y[:n], fs, trim=False)
    return {"syncs": len(r["syncs"]), "eye": r["eye"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-f", "--center", type=float, required=True,
                    help="SDR center frequency, Hz")
    ap.add_argument("-s", "--rate", type=float, required=True,
                    help="SDR sample rate, Hz (use a multiple of --wav-rate)")
    ap.add_argument("-c", "--channel", type=float,
                    help="channel to camp on, Hz (default: --center)")
    ap.add_argument("-o", "--out", help="output WAV path")
    ap.add_argument("--tcp", type=int, nargs="?", const=7355,
                    help="also serve the stream on this TCP port for "
                         "`dsd-fme -i tcp` (default port 7355)")
    ap.add_argument("--tcp-host", default="127.0.0.1",
                    help="address to listen on (default: %(default)s)")
    ap.add_argument("--wav-rate", type=float, default=DISC_RATE,
                    help="output sample rate, Hz (default: %(default)s)")
    ap.add_argument("--keep-hz", type=float, default=KEEP_HZ,
                    help="channel half-bandwidth, Hz")
    ap.add_argument("--input", help="recorded cs8 capture instead of stdin")
    ap.add_argument("--pace", action="store_true",
                    help="read --input at real time, as a stand-in for a radio")
    ap.add_argument("--duration", type=float, help="stop after this many seconds")
    ap.add_argument("--block", type=float, default=2.0,
                    help="seconds of raw samples per processing block")
    ap.add_argument("--no-dc-block", dest="dc_block", action="store_false",
                    help="do not subtract the receiver's DC offset")
    ap.add_argument("--probe", type=float, default=10.0,
                    help="seconds between DMR health probes, 0 to disable")
    args = ap.parse_args(argv)
    if not (args.out or args.tcp):
        ap.error("nothing to write to: give --out, --tcp, or both")

    fs, fc = args.rate, args.center
    ch = args.channel if args.channel else fc
    ch = ch * 1e6 if ch < 1e6 else ch
    d, block, overlap = plan(fs, args.wav_rate, args.block)
    if abs(ch - fc) < 100e3:
        sys.stderr.write(
            f"# WARNING: {abs(ch-fc)/1e3:.1f} kHz from center. A receiver leaks "
            f"its local oscillator\n#   into 0 Hz of baseband, and this channel "
            f"is sitting on top of it. Tune\n#   the SDR a few hundred kHz away "
            f"and camp on the same channel:\n"
            f"#   -f {(ch - 500e3)/1e6:.3f}e6 ... -c {ch/1e6:.4f}e6\n")

    stats = {"dropped": 0, "worst_gap": 0.0, "tcp_dropped": 0}
    src = (file_blocks(args.input, block * 2, block / fs if args.pace else 0.0)
           if args.input else stdin_blocks(block * 2, stats))

    sys.stderr.write(
        f"# camped on {ch/1e6:.4f} MHz ({(ch-fc)/1e3:+.1f} kHz from center), "
        f"{fs/1e6:.2f} Msps / {d} = {args.wav_rate/1e3:.0f} kHz\n"
        f"# {block/fs:.2f}s blocks, {overlap/fs*1000:.0f} ms overlap, "
        f"{args.wav_rate*2/1e3:.0f} kB/s\n")

    sinks = []
    if args.out:
        sinks.append(WavWriter(args.out, args.wav_rate))
        sys.stderr.write(f"# writing {args.out}\n")
    tcp = None
    if args.tcp:
        tcp = TcpSink(args.tcp, args.tcp_host, stats=stats)
        sinks.append(tcp)
        sys.stderr.write(
            f"# listening on {args.tcp_host}:{args.tcp} -- decode live with:\n"
            f"#   dsd-fme -fs -i tcp:{args.tcp_host}:{args.tcp} -o pulse -P -7 decoded\n")
    t0 = time.time()
    t_work = 0.0
    written = 0.0
    next_probe = 0.0
    try:
        for f, y, cfs, fe in channel_stream(src, fs, ch - fc, args.wav_rate,
                                            args.keep_hz, d, block, overlap,
                                            args.dc_block):
            tick = time.perf_counter()
            pcm = pcm16(scale_discriminator(f))
            for sink in sinks:
                sink.write(pcm)
            written += f.size / args.wav_rate
            t_work += time.perf_counter() - tick

            if args.probe and written >= next_probe:
                next_probe = written + args.probe
                p = probe(y, cfs)
                wall = max(time.time() - t0, 1e-9)
                sys.stderr.write(
                    f"# {written:7.0f}s written, {wall:7.0f}s wall, "
                    f"x{written/wall:.2f}, cpu {t_work/wall:.2f}, "
                    f"drop {stats['dropped']}, "
                    f"gap {stats['worst_gap']*1000:.0f} ms | "
                    + (f"syncs {p['syncs']:3d} eye {p['eye']:.2f} "
                       if p else "")
                    + f"dc {float(np.mean(f)):+.0f} Hz | "
                    + f"adc rms {fe[0]:.0f}/127 clip {fe[1]*100:.1f}% "
                    + f"dcoff {fe[2]:+d}{fe[3]:+d}"
                    + (f" | tcp {tcp.count()} client(s), "
                       f"{stats['tcp_dropped']} dropped" if tcp else "")
                    + "\n")
            if args.duration and written >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        for sink in sinks:
            sink.close()
        sys.stderr.write(
            f"# streamed {written:.1f}s, drop {stats['dropped']}, "
            f"worst read gap {stats['worst_gap']*1000:.0f} ms"
            + (f", tcp dropped {stats['tcp_dropped']}" if tcp else "") + "\n")
        if args.out:
            sys.stderr.write(
                f"# decode with: dsd-fme -fs -i {args.out} -o pulse -P -7 decoded\n")


if __name__ == "__main__":
    main()
