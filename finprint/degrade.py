"""Controlled audio degradations, for measuring what survives a real demo.

Everything the model was trained on is a clean Watkins archive recording. A demo
often is not: a compressed download, a call played through a laptop speaker and
caught on a phone, a hydrophone in a noisy sea. `scripts/robustness.py` runs the
held-out split through each transform below and reports what the app would say,
so the limits printed in reports/error_analysis.md are measured rather than
guessed at.

These are *simulations*, and the honest caveat is that they are crude: a real
speaker and a real room have a response this does not reproduce. They are useful
because they are reproducible and they bracket the right failure — not because
they are faithful. Nothing here is imported at serve time.

Every transform is deterministic given the `rng` it is handed, so a reported
number can be reproduced exactly rather than being resampled on each run.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import config as C
from .audio import loudest_window
from .features import trim_silence


def signal_rms(wav: np.ndarray) -> float:
    """RMS of the call itself — the reference level for SNR.

    Deliberately not the whole-clip RMS. WMMS clips run up to ~114 s and are
    often mostly quiet sea, so whole-clip RMS is dominated by the silence: "0 dB
    SNR" would then mean something far gentler on a long clip than on a short
    one, and the worst row of the robustness table would be reporting a
    different experiment per clip.

    The loudest analysis window alone is not enough either — the median clip's
    call is ~1.7 s inside a 4 s window, so three-quarters of that reference can
    still be silence. Trimming first is what makes the number describe the call.
    """
    wav = np.asarray(wav, dtype=np.float32).ravel()
    seg = loudest_window(trim_silence(wav, top_db=30.0), C.ANALYSIS_SAMPLES)
    if seg.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))


def add_noise(wav: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise at a given SNR against the call itself."""
    wav = np.asarray(wav, dtype=np.float32).ravel()
    sig = signal_rms(wav)
    if sig <= 0:
        return wav
    noise_rms = sig / (10.0 ** (snr_db / 20.0))
    noise = rng.normal(0.0, noise_rms, size=wav.shape)
    return (wav + noise).astype(np.float32)


def band_limit(wav: np.ndarray, low_hz: float, high_hz: float,
               sr: int = C.SAMPLE_RATE) -> np.ndarray:
    """Zero-phase Butterworth bandpass — what a phone mic and codec throw away."""
    from scipy.signal import butter, sosfiltfilt

    wav = np.asarray(wav, dtype=np.float32).ravel()
    nyq = sr / 2.0
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 0.999)
    if low >= high:
        return wav
    sos = butter(4, [low, high], btype="band", output="sos")
    # filtfilt needs a few times the filter length; a very short clip is left
    # alone rather than raising.
    if len(wav) <= 3 * sos.shape[0] * 2:
        return wav
    return sosfiltfilt(sos, wav.astype(np.float64)).astype(np.float32)


def reverb(wav: np.ndarray, sr: int = C.SAMPLE_RATE, decay_s: float = 0.35,
           wet: float = 0.35, *, rng: np.random.Generator) -> np.ndarray:
    """Convolve with an exponentially-decaying noise burst — a crude small room.

    Not a measured impulse response. It smears transients across time, which is
    the part of playing-a-call-out-loud that actually costs the model accuracy
    (echolocation clicks stop being clicks), and that is all it is claiming to do.
    """
    wav = np.asarray(wav, dtype=np.float32).ravel()
    n = int(sr * decay_s)
    if n < 2 or wav.size == 0:
        return wav
    t = np.arange(n) / sr
    ir = rng.normal(0.0, 1.0, n) * np.exp(-t / (decay_s / 3.0))
    ir[0] += 1.0 / max(wet, 1e-6)          # keep the direct path dominant
    ir /= np.sqrt(np.sum(ir ** 2))
    wet_sig = np.convolve(wav, ir, mode="full")[:len(wav)]
    return wet_sig.astype(np.float32)


def mp3_roundtrip(wav: np.ndarray, bitrate_kbps: int,
                  sr: int = C.SAMPLE_RATE) -> np.ndarray:
    """Encode to mp3 at `bitrate_kbps` and decode back, via ffmpeg.

    Returns the clip re-decoded at `sr`. Raises if ffmpeg is unavailable, rather
    than silently reporting an undegraded clip as compressed.
    """
    import soundfile as sf

    wav = np.asarray(wav, dtype=np.float32).ravel()
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "in.wav"
        enc = Path(d) / "out.mp3"
        dec = Path(d) / "back.wav"
        sf.write(src, wav, sr)
        for cmd in (
            ["ffmpeg", "-v", "error", "-y", "-i", str(src),
             "-b:a", f"{bitrate_kbps}k", "-ar", str(sr), "-ac", "1", str(enc)],
            ["ffmpeg", "-v", "error", "-y", "-i", str(enc),
             "-ar", str(sr), "-ac", "1", str(dec)],
        ):
            subprocess.run(cmd, check=True, capture_output=True)
        out, _ = sf.read(dec, dtype="float32", always_2d=False)
    return np.asarray(out, dtype=np.float32).ravel()


def speaker_playback(wav: np.ndarray, sr: int = C.SAMPLE_RATE, *,
                     rng: np.random.Generator) -> np.ndarray:
    """Play it through a laptop speaker, record it on a phone across the room.

    Band-limiting (neither end of the spectrum survives the trip), room reverb,
    and the room's own noise floor, in that order — the order they happen in.
    """
    out = band_limit(wav, 300.0, 8000.0, sr)
    out = reverb(out, sr, rng=rng)
    return add_noise(out, 15.0, rng)


# The conditions the robustness report walks, in the order it prints them. Each
# is a callable of (wav, rng); `clean` is the baseline every other row is read
# against, so it stays first.
CONDITIONS: dict[str, callable] = {
    "clean": lambda w, rng: w,
    "mp3 64 kbps": lambda w, rng: mp3_roundtrip(w, 64),
    "mp3 32 kbps": lambda w, rng: mp3_roundtrip(w, 32),
    "band-limited 300 Hz-8 kHz": lambda w, rng: band_limit(w, 300.0, 8000.0),
    "noise 20 dB SNR": lambda w, rng: add_noise(w, 20.0, rng),
    "noise 10 dB SNR": lambda w, rng: add_noise(w, 10.0, rng),
    "noise 0 dB SNR": lambda w, rng: add_noise(w, 0.0, rng),
    "speaker playback + phone capture": lambda w, rng: speaker_playback(w, rng=rng),
}
