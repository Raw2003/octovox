"""WAV metadata helpers.

Reads channel count, sample rate, duration, peak and RMS levels from a WAV
file. Mirrors the behaviour of the original ``app.wav_info`` exactly so the
frontend keeps receiving the same JSON shape.
"""
import numpy as np
from scipy.io import wavfile


def wav_info(path):
    """Return a metadata dict for a WAV file, with safe fallbacks on error."""
    try:
        fs, data = wavfile.read(str(path))
        ch = data.shape[1] if data.ndim > 1 else 1
        dur = data.shape[0] / fs
        peak = float(np.max(np.abs(data)) / (np.iinfo(data.dtype).max
                if np.issubdtype(data.dtype, np.integer) else 1.0))
        rms = float(np.sqrt(np.mean((data.astype(np.float64)/(np.iinfo(data.dtype).max
                if np.issubdtype(data.dtype, np.integer) else 1.0))**2)))
        rms_db = 20*np.log10(rms + 1e-12)
        peak_dbfs = 20*np.log10(peak + 1e-12)
        st = path.stat()
        return {"name": path.name, "channels": int(ch), "samplerate": int(fs),
                "duration": float(dur), "peak": peak, "rms_db": float(rms_db),
                "peak_dbfs": float(peak_dbfs),
                "size_kb": st.st_size // 1024,
                "mtime": float(st.st_mtime)}
    except Exception:
        st = path.stat() if path.exists() else None
        return {"name": path.name, "channels": 0, "samplerate": 0,
                "duration": 0, "peak": 0, "rms_db": -120,
                "peak_dbfs": -120.0,
                "size_kb": (st.st_size // 1024) if st else 0,
                "mtime": (st.st_mtime if st else 0.0)}
