import argparse
import collections
import glob
import os
import re
from fractions import Fraction

import numpy as np
from scipy import signal

from frs_scan.dmr import DEV_OUTER, demod_dmr, discriminator
from frs_scan.iq import load_iq, write_wav
from frs_scan.nfm import CHAN_RATE

# dsd-fme reads 48 kHz mono WAVs of the FM-demodulated signal
DISC_RATE = 48000

# A burst is 288 bits: CACH(24) info(98) slot type(10) sync(48) slot type(10) info(98)
INFO_A = slice(24, 122)
INFO_B = slice(190, 288)
INFO_BITS = 196

# The idle payload is a constant of the protocol: these 196 bits are byte-identical 
# from thirteen different color codes. Bursts are idle if they match.
IDLE_INFO = bytes.fromhex("53c25eaba8671dc7383bd9363f6e465171b48ca6d4fc610b40")

# A voice burst carries three 72-bit AMBE+2 frames instead of a BPTC payload
VOICE_1 = slice(24, 96)
VOICE_2A = slice(96, 132)
VOICE_2B = slice(180, 216)
VOICE_3 = slice(216, 288)
AMBE_FRAME_BITS = 72
AMBE_FRAME_MS = 20

# Bursts A to F of a voice superframe, and the six bursts of the other TDMA
# slot between them
SUPERFRAME_BURSTS = 6
SLOT_STRIDE = 2

# Distance to the idle pattern is sharply bimodal. Anywhere in the gap works
IDLE_MAX_ERRORS = 20


def _bits(dibits):
    """144 dibits -> 288 bits."""
    b = dibits.astype(np.uint8).reshape(-1, 1) << 6
    return np.unpackbits(b, axis=1)[:, :2].ravel()


def info_field(dibits):
    """The 196 information bits of a burst."""
    if dibits.size < 144:
        return None
    b = _bits(dibits)
    return np.concatenate((b[INFO_A], b[INFO_B]))


def voice_frames(dibits):
    """Three AMBE+2 frames from a voice burst, as a 3x72 bit array."""
    if dibits.size < 144:
        return None
    b = _bits(dibits)
    return np.stack((b[VOICE_1],
                     np.concatenate((b[VOICE_2A], b[VOICE_2B])),
                     b[VOICE_3]))


def voice_superframes(bursts):
    """
    Runs of six voice bursts, one per sync that says voice.

    Only burst A of a superframe carries a sync; B to F replace it with an
    EMB, so they cannot be found by matching and have to be counted off
    instead. The repeater interleaves the two TDMA slots burst by burst, so
    the rest of this superframe is at +2, +4 ... +10, and the odd bursts
    between them belong to the other slot.
    """
    out = []
    for i, b in enumerate(bursts):
        if not (b["sync"] and b["sync"].endswith("voice")):
            continue
        idx = range(i, i + SLOT_STRIDE * SUPERFRAME_BURSTS, SLOT_STRIDE)
        if idx[-1] >= len(bursts):
            continue
        # a data sync in the run means the call ended and the slot moved on
        run = [bursts[j] for j in idx]
        if any(r["sync"] and r["sync"].endswith("data") for r in run):
            continue
        out.append(run)
    return out


def write_ambe(path, frames):
    """
    AMBE+2 frames

    AMBE+2 is a proprietary vocoder, so turning these frames into audio needs a 
    decoder implementation. dsd-fme will not read them -- it wants the WAV from 
    discriminator_wav and does its own demodulation -- so these are for md380-emu, 
    which takes frames directly, after the 72-bit on-air frames are reduced to 
    the 49 bits of voice parameters inside them.
    """
    packed = np.packbits(np.asarray(frames, np.uint8).reshape(-1, AMBE_FRAME_BITS),
                         axis=1)
    with open(path, "wb") as f:
        f.write(packed.tobytes())
    return packed.shape[0]


def discriminator_wav(y, fs, inverted=False, rate=DISC_RATE):
    """
    Channelized baseband -> the FM-demodulated WAV dsd-fme expects.

    No de-emphasis and no audio band limit, which would flatten the
    4FSK symbol levels into each other.

    Scaling is by deviation, so a burst of noise between transmissions cannot 
    decide the level of the signal. The outer symbols of a correctly tuned DMR 
    carrier land at half scale whatever else is in the file. `inverted` re-mirrors 
    a capture where the sideband came out the wrong way.
    """
    f = discriminator(y, fs)
    if inverted:
        f = -f
    ratio = Fraction(rate / fs).limit_denominator(1000)
    f = signal.resample_poly(f, ratio.numerator, ratio.denominator)
    return scale_discriminator(f)


def scale_discriminator(f):
    """
    Deviation in Hz -> WAV sample values

    Kept apart from discriminator_wav so camp.py can scale a stream the same
    way.
    """
    return np.clip(f / (2.0 * DEV_OUTER), -1.0, 1.0).astype(np.float32)


def idle_errors(info):
    """Bit distance from a burst's info field to the idle pattern."""
    ref = np.unpackbits(np.frombuffer(IDLE_INFO, np.uint8))[:INFO_BITS]
    return int(np.count_nonzero(info != ref))


def is_idle(info, max_errors=IDLE_MAX_ERRORS):
    return idle_errors(info) <= max_errors


# ENCRYPTION

def encryption_status(bursts):
    """
    Check for encryption signalling. *Does not decrypt anything*.

    DMR announces an encrypted call with a privacy indicator header. 
    A call with no PI header does not signal encryption. It is still possible
    that encryption is applied but not announced.
    """
    pi = [b for b in bursts
          if b.get("data_type") == 0 and b.get("info") is not None
          and not is_idle(b["info"])]
    if pi:
        return {"verdict": "encrypted",
                "evidence": f"{len(pi)} privacy indicator header burst(s)",
                "algorithm": None,
                "note": "algorithm and key id are in the PI payload, not decoded"}
    return {"verdict": "no encryption signalled",
            "evidence": "no privacy indicator header",
            "algorithm": None,
            "note": "absence of signalling is not proof of clear traffic"}


# PER CAPTURE

def analyse(path, rate=CHAN_RATE, max_errors=IDLE_MAX_ERRORS):
    """One capture -> its DMR content."""
    y = load_iq(path, "cs8")
    r = demod_dmr(y, rate)
    out = {"path": path, "eye": r["eye"], "bursts": len(r["bursts"]),
           "syncs": len(r["syncs"]), "dmr": bool(r["syncs"]),
           "inverted": r["inverted"], "_iq": y, "_rate": rate}
    if not r["syncs"]:
        return out

    for b in r["bursts"]:
        b["info"] = info_field(b["dibits"])

    data = [b for b in r["bursts"] if b["sync"] and b["sync"].endswith("data")]
    voice = [b for b in r["bursts"] if b["sync"] and b["sync"].endswith("voice")]
    idle = [b for b in data if b["info"] is not None and is_idle(b["info"], max_errors)]
    traffic = [b for b in data if b["info"] is not None
               and not is_idle(b["info"], max_errors)]

    # color code is read without checking its Golay parity, so trust it only
    # where the bursts agree with each other.
    ccs = collections.Counter(b["color_code"] for b in data
                              if b.get("color_code") is not None)
    cc, cc_n = (ccs.most_common(1)[0] if ccs else (None, 0))

    out.update({
        "idle": len(idle), "traffic": len(traffic), "voice": len(voice),
        "color_code": cc,
        "color_code_agreement": cc_n / len(data) if data else 0.0,
        "payloads": collections.Counter(np.packbits(b["info"]).tobytes()
                                        for b in traffic),
        "superframes": voice_superframes(r["bursts"]),
        "encryption": encryption_status(r["bursts"]),
        "_bursts": r["bursts"],
    })
    return out


def is_interesting(a):
    return bool(a.get("dmr")) and (a.get("traffic") or a.get("voice"))


def save_discriminator(a, wav_dir):
    """
    Write the capture as a 48 kHz WAV for dsd-fme, whatever it contains.

    Not restricted to captures with voice: dsd-fme reads the signalling too.
    """
    if not a.get("dmr"):
        return None
    stem = os.path.splitext(os.path.basename(a["path"]))[0]
    path = os.path.join(wav_dir, stem + "_disc.wav")
    os.makedirs(wav_dir, exist_ok=True)
    write_wav(path, discriminator_wav(a["_iq"], a["_rate"], a["inverted"]),
              DISC_RATE)
    return path


def save_voice(a, audio_dir):
    """
    Write the AMBE frames of an unencrypted capture.

    The filename carries a `_digital-voice` suffix, so the frames are
    identifiable as AMBE+2 from the name alone.
    """
    sf = a.get("superframes") or []
    if not sf or a["encryption"]["verdict"] == "encrypted":
        return None
    frames = [f for run in sf for b in run
              for f in voice_frames(b["dibits"])]
    if not frames:
        return None
    stem = os.path.splitext(os.path.basename(a["path"]))[0]
    path = os.path.join(audio_dir, stem + "_digital-voice.ambe")
    os.makedirs(audio_dir, exist_ok=True)
    write_ambe(path, frames)
    return path


# REPORTING

def describe(a):
    name = os.path.basename(a["path"])
    freq = re.search(r"_([\d.]+)MHz", name)
    lines = [name]
    head = f"  {a['bursts']} bursts, eye {a['eye']:.2f}"
    if a.get("color_code") is not None:
        head += (f", color code {a['color_code']}"
                 f" ({a['color_code_agreement']*100:.0f}% agreement)")
    lines.append(head)
    lines.append(f"  {a['idle'] + a['traffic']} data bursts: {a['idle']} idle, "
                 f"{a['traffic']} traffic")
    sf = a.get("superframes") or []
    if sf:
        secs = len(sf) * SUPERFRAME_BURSTS * 3 * AMBE_FRAME_MS / 1000.0
        lines.append(f"  {len(sf)} voice superframes = {secs:.2f}s of AMBE+2")
    if a["payloads"]:
        lines.append(f"  {len(a['payloads'])} distinct traffic payload(s):")
        for pl, n in a["payloads"].most_common(4):
            lines.append(f"    x{n:<4d} {pl.hex()[:32]}...")
    e = a["encryption"]
    lines.append(f"  encryption: {e['verdict']} -- {e['evidence']}")
    if e["verdict"] == "encrypted":
        lines.append(f"              {e['note']}")
    return "\n".join(lines)


def debug_slot_types(a):
    """Raw 20-bit slot type per burst."""
    rows = []
    for b in a.get("_bursts", []):
        if b.get("color_code") is None:
            continue
        rows.append(f"    cc={b['color_code']:2d} dt={b['data_type']:2d} "
                    f"{b['data_type_name']}")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("path", nargs="?", default="hits",
                    help="directory of captures, or a single .iq file")
    ap.add_argument("--pattern", default="*_digital.iq",
                    help="glob within a directory (default: %(default)s)")
    ap.add_argument("--rate", type=float, default=CHAN_RATE,
                    help="sample rate of the saved IQ, Hz")
    ap.add_argument("--max-idle-errors", type=int, default=IDLE_MAX_ERRORS,
                    help="bit errors still counted as an idle burst")
    ap.add_argument("--audio-dir",
                    help="write AMBE+2 frames of unencrypted voice here")
    ap.add_argument("--wav-dir",
                    help="write 48 kHz FM-demodulated WAVs for dsd-fme here")
    ap.add_argument("--all", action="store_true",
                    help="report every capture, not just the ones with traffic")
    ap.add_argument("--debug", action="store_true",
                    help="dump the slot type of every burst")
    args = ap.parse_args(argv)

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, args.pattern)))
    else:
        files = [args.path]
    if not files:
        raise SystemExit(f"no captures matching {args.pattern} in {args.path}")

    n_dmr = n_shown = 0
    enc, written, wavs = [], [], []
    for p in files:
        a = analyse(p, args.rate, args.max_idle_errors)
        n_dmr += bool(a["dmr"])
        if a["dmr"] and a["encryption"]["verdict"] == "encrypted":
            enc.append(os.path.basename(p))
        if not (args.all or is_interesting(a)):
            continue
        n_shown += 1
        if args.audio_dir:
            w = save_voice(a, args.audio_dir)
            if w:
                written.append(w)
        if args.wav_dir:
            w = save_discriminator(a, args.wav_dir)
            if w:
                wavs.append(w)
        print(describe(a) if a["dmr"] else
              f"{os.path.basename(p)}\n  no DMR sync found")
        if args.debug:
            print(debug_slot_types(a))
        print()

    print(f"{len(files)} captures, {n_dmr} with DMR sync, "
          f"{n_shown} reported.")
    if enc:
        print(f"encryption signalled in {len(enc)}: {', '.join(enc)}")
    if wavs:
        print(f"wrote {len(wavs)} discriminator WAV(s) -- decode with:\n"
              f"  dsd-fme -fs -i {wavs[0]} -o null -P")
    if written:
        n = sum(os.path.getsize(w) for w in written) // 9
        print(f"wrote {len(written)} AMBE file(s), {n} frames = "
              f"{n * AMBE_FRAME_MS / 1000:.1f}s (for md380-emu; dsd-fme "
              f"wants the --wav-dir output instead)")


if __name__ == "__main__":
    main()
