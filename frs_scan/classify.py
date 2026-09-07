import numpy as np
from scipy import signal

from frs_scan.nfm import AUDIO_RATE, DEVIATION, burst_body

# ---------------------------------------------------------------- features

def carrier_stats(y, fs, deviation=DEVIATION, body=None):
    """
    Checks if there is a carrier, and if it is multi-level.

    Both are stationary statistics of the modulation, so a short window from
    the middle of a burst is sufficient.
    """
    env = np.abs(y)
    if body is None:
        body = burst_body(env, fs)

    # Envelope constancy -- an FM carrier limits to near-constant amplitude
    b = env[body]
    env_cv = float(b.std() / (b.mean() + 1e-20))

    d = np.angle(y[1:] * np.conj(y[:-1])).astype(np.float32)
    d *= fs / (2 * np.pi * deviation)

    # excess kurtosis of the instantaneous frequency
    dm = d[body] if body.stop <= d.size else d
    z = dm - dm.mean()
    var = float(np.mean(z ** 2))
    kurt = float(np.mean(z ** 4) / (var ** 2 + 1e-20) - 3.0) if var > 0 else 0.0
    return {"env_cv": env_cv, "kurt": kurt}


def spectral_flatness(x, fs, lo=300.0, hi=3400.0):
    """Geometric/arithmetic mean of the PSD in a band. 1.0 = white noise."""
    if x.size < 512:
        return 1.0
    f, p = signal.welch(x, fs, nperseg=min(1024, x.size))
    sel = (f >= lo) & (f <= hi)
    p = p[sel] + 1e-20
    if p.size < 8:
        return 1.0
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def voiced_fraction(audio, fs, lo_hz=80.0, hi_hz=300.0, peak=0.35, win_s=0.04):
    """
    Share of audio frames carrying a clear pitch.

    Speech is periodic over 40 ms frames, so its normalised autocorrelation
    peaks at a lag inside the human pitch range. Noise has no such peak.
    """
    w = int(win_s * fs)
    hop = w // 2
    # three or four frames quantize the fraction to quarters
    if audio.size < max(2 * w, int(0.3 * fs)):
        return float("nan")
    n = 1 + (audio.size - w) // hop
    idx = np.arange(w)[None, :] + hop * np.arange(n)[:, None]
    frames = audio[idx].astype(np.float32)
    # a frequency-offset carrier's DC step correlates with itself at every lag, so calculate per frame
    frames -= frames.mean(axis=1, keepdims=True)
    frames *= np.hanning(w).astype(np.float32)
    nfft = 1 << int(np.ceil(np.log2(2 * w)))
    # wiener-khinchin
    ac = np.fft.irfft(np.abs(np.fft.rfft(frames, nfft, axis=1)) ** 2, nfft, axis=1)
    ac = ac[:, :w] / (ac[:, :1] + 1e-20)
    lags = slice(int(fs / hi_hz), int(fs / lo_hz))
    if lags.stop <= lags.start:
        return float("nan")
    return float(np.mean(ac[:, lags].max(axis=1) > peak))


def tone_fraction(audio, fs, lo=200.0, hi=3800.0, width=100.0):
    """
    Share of audio power within +/-width of the single strongest peak.

    Tells a keyed CW tone from speech.
    """
    if audio.size < 512:
        return 0.0
    f, p = signal.welch(audio, fs, nperseg=min(2048, audio.size))
    sel = (f >= lo) & (f <= hi)
    f, p = f[sel], p[sel]
    if p.size < 8:
        return 0.0
    near = np.abs(f - f[p.argmax()]) <= width
    return float(p[near].sum() / (p.sum() + 1e-20))


def syllabic_ratio(audio, fs):
    """
    Share of the audio envelope's modulation energy at speech rates.

    Speech switches between syllables a few times a second, steady noise and
    steady data are flat.

    Result is a ratio. NaN when there is not enough audio to see 2 Hz
    """
    if audio.size < int(1.5 * fs):
        return float("nan")
    dec = max(1, int(fs / 200.0))
    env = np.abs(signal.lfilter(np.ones(dec) / dec, 1.0, np.abs(audio))[::dec])
    fs_e = fs / dec
    if env.size < 128:
        return float("nan")
    env = env - env.mean()
    f, p = signal.welch(env, fs_e, nperseg=min(256, env.size))
    band = p[(f >= 2.0) & (f <= 8.0)].sum()
    total = p[(f >= 0.5) & (f <= 20.0)].sum()
    return float(band / (total + 1e-20))


def features(y, fs, audio, audio_rate=AUDIO_RATE, deviation=DEVIATION):
    """
    All features from one burst.

    Takes both the complex baseband and the audio demodulated from it.
    """
    body = burst_body(np.abs(y), fs)
    feats = carrier_stats(y, fs, deviation, body)
    r = audio_rate / fs
    ab = audio[int(body.start * r):max(int(body.stop * r), int(body.start * r) + 1)]
    feats["flatness"] = spectral_flatness(ab, audio_rate)
    feats["syllabic"] = syllabic_ratio(ab, audio_rate)
    feats["voiced"] = voiced_fraction(ab, audio_rate)
    feats["tone"] = tone_fraction(ab, audio_rate)
    return feats


# CUTOFFS

CV_CUT = 0.35 # envelope std/mean above this -> no carrier
DIGITAL_CUT = -0.2 # excess kurtosis below this -> multi-level FSK
VOICED_CUT = 0.35 # share of frames with a clear pitch peak
TONE_CUT = 0.85 # audio power share within +/-100 Hz of the strongest peak
FLAT_MIN = 0.05 # audio flatness below this -> a bare tone
SYLLABIC_CUT = 0.40 



ALL_KINDS = ("voice", "morse", "digital", "noise")


def prescreen(feats, cv_cut=CV_CUT):
    """
    Short probe verdict, or None if it needs the full demod.

    Only the envelope test is safe here, an FM carrier limits to constant amplitude for as long as it is up.
    """
    if feats["env_cv"] > cv_cut:
        return "noise"
    return None


def classify(feats, digital_cut=DIGITAL_CUT, cv_cut=CV_CUT,
             flat_min=FLAT_MIN, voiced_cut=VOICED_CUT, tone_cut=TONE_CUT):
    """
    Four checks: is there a carrier, is it data, is it a keyed tone, and does it sound like a voice.
    """
    early = prescreen(feats, cv_cut)
    if early:
        return early
    if feats["kurt"] < digital_cut:
        return "digital"
    if feats.get("tone", 0.0) > tone_cut:
        return "morse"
    voiced = feats.get("voiced", float("nan"))
    if np.isnan(voiced) or voiced < voiced_cut:
        return "noise"
    # voiced with no spectral spread is a tone
    if feats["flatness"] < flat_min:
        return "noise"
    return "voice"


def keep_set(name):
    """
    --keep choice -> the set of classify() results worth writing
    """
    shorthand = {"voice": {"voice", "morse"},
                 "speech": {"voice"},
                 "signal": {"voice", "morse", "digital"},
                 "all": set(ALL_KINDS)}
    if name in shorthand:
        return shorthand[name]
    kinds = {k.strip() for k in name.split(",") if k.strip()}
    bad = kinds - set(ALL_KINDS)
    if bad or not kinds:
        raise ValueError(f"unknown kind(s): {', '.join(sorted(bad)) or name}")
    return kinds


def add_classifier_args(ap):
    """Add the five threshold overrides to an ArgumentParser"""
    ap.add_argument("--digital-cut", type=float, default=DIGITAL_CUT,
                    help="kurtosis below which a burst is called digital")
    ap.add_argument("--cv-cut", type=float, default=CV_CUT,
                    help="envelope std/mean above which there is no carrier")
    ap.add_argument("--flat-min", type=float, default=FLAT_MIN,
                    help="audio spectral flatness below which it is not speech")
    ap.add_argument("--voiced-cut", type=float, default=VOICED_CUT,
                    help="minimum share of audio frames with a clear pitch")
    ap.add_argument("--tone-cut", type=float, default=TONE_CUT,
                    help="audio power share in one peak above which it is morse")
    return ap


def cut_kwargs(args):
    """The parsed threshold overrides, as classify()/prescreen() keywords."""
    return dict(digital_cut=args.digital_cut, cv_cut=args.cv_cut,
                flat_min=args.flat_min, voiced_cut=args.voiced_cut,
                tone_cut=args.tone_cut)
