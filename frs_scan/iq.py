import os
import wave

import numpy as np

# sample format on disk
FORMATS = {
    "cs8":  (np.int8,    127.0),
    "cs16": (np.int16, 32767.0),
    "cf32": (np.float32,   1.0),
}

RAW = np.int8 # hackrf adc is 8 bit


def load_iq(path: str, fmt: str, offset_samples: int = 0, count: int | None = None) -> np.ndarray:
    """Read interleaved IQ from disk as complex64."""
    dtype, scale = FORMATS[fmt]
    itemsize = np.dtype(dtype).itemsize
    raw = np.fromfile(path, dtype=dtype,
                      offset=offset_samples * 2 * itemsize,
                      count=-1 if count is None else count * 2)
    if raw.size % 2:
        raw = raw[:-1]
    iq = raw.astype(np.float32).view()
    return (iq[0::2] + 1j * iq[1::2]).astype(np.complex64) / scale


def file_length(path: str, fmt: str) -> int:
    dtype, _ = FORMATS[fmt]
    return os.path.getsize(path) // (2 * np.dtype(dtype).itemsize)


def write_cs8(path: str, x: np.ndarray) -> None:
    """Complex baseband -> interleaved int8, scaled to leave headroom."""
    peak = max(float(np.max(np.abs(x.real))), float(np.max(np.abs(x.imag))), 1.0)
    scale = 100.0 / peak
    out = np.empty(x.size * 2, dtype=RAW)
    out[0::2] = np.clip(np.real(x) * scale, -127, 127)
    out[1::2] = np.clip(np.imag(x) * scale, -127, 127)
    out.tofile(path)


def write_wav(path: str, audio: np.ndarray, rate: float) -> None:
    """Mono 16-bit WAV, peak-normalized to 0.89 of full scale."""
    peak = np.max(np.abs(audio)) or 1.0
    pcm = np.clip(audio / peak * 0.89, -1, 1)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(pcm.tobytes())
