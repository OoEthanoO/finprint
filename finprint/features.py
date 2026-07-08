"""Acoustic feature extraction for a single call.

Turns a mono waveform into the interpretable numbers a bioacoustician reads off
a spectrogram: dominant frequency, bandwidth, tonality, pitch (f0) contour, and
pulse-repetition rate. These feed the call-type classifier and are shown in the
app so a prediction is explainable, not a black box.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import librosa
import numpy as np

from . import config as C

# f0 search range for marine-mammal tonal sounds (Hz).
_F0_MIN = 40.0
_F0_MAX = min(4000.0, C.SAMPLE_RATE / 2 - 100)


@dataclass
class AcousticFeatures:
    duration_s: float
    dominant_freq_hz: float
    spectral_centroid_hz: float
    bandwidth_hz: float
    low_freq_hz: float          # 5th-percentile energy edge
    high_freq_hz: float         # 95th-percentile energy edge
    spectral_flatness: float    # 0 = pure tone, 1 = white noise
    zero_crossing_rate: float
    f0_mean_hz: float           # NaN-safe: 0.0 when unvoiced
    f0_range_hz: float
    voiced_fraction: float      # share of frames with a detectable pitch
    pulse_rate_hz: float        # repetition rate of the amplitude envelope

    def as_dict(self) -> dict:
        return asdict(self)


def _energy_edges(mag: np.ndarray, freqs: np.ndarray) -> tuple[float, float]:
    """5th / 95th percentile of the cumulative energy across frequency."""
    spec = mag.mean(axis=1)
    total = spec.sum()
    if total <= 0:
        return 0.0, 0.0
    cdf = np.cumsum(spec) / total
    lo = float(freqs[np.searchsorted(cdf, 0.05)])
    hi = float(freqs[min(np.searchsorted(cdf, 0.95), len(freqs) - 1)])
    return lo, hi


def _pulse_rate(wav: np.ndarray, sr: int) -> float:
    """Estimate envelope repetition rate via autocorrelation (Hz), or 0 if none.

    We autocorrelate the amplitude envelope but *skip the main lobe* (the trivial
    self-similarity near lag 0) and only accept a strong, genuine secondary peak.
    Without that, tonal calls falsely read as maximally pulsed.
    """
    env = np.abs(wav)
    win = max(1, int(sr * 0.001))
    if win > 1:
        env = np.convolve(env, np.ones(win) / win, mode="same")
    env = env - env.mean()
    if np.allclose(env, 0):
        return 0.0
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]

    # end of the main lobe = first lag where autocorrelation crosses zero
    below = np.where(ac <= 0.0)[0]
    main_lobe_end = int(below[0]) if below.size else max(1, int(sr / 2000))

    lo_lag = max(main_lobe_end, int(sr / 2000))   # up to 2000 pulses/s
    hi_lag = min(len(ac) - 1, int(sr / 10))       # down to 10 pulses/s
    if hi_lag <= lo_lag:
        return 0.0
    seg = ac[lo_lag:hi_lag]
    peak = int(np.argmax(seg)) + lo_lag
    if ac[peak] < 0.4:                             # require clear periodicity
        return 0.0
    return float(sr / peak)


def extract(wav: np.ndarray, sr: int = C.SAMPLE_RATE) -> AcousticFeatures:
    wav = np.asarray(wav, dtype=np.float32).ravel()
    wav = wav - float(wav.mean())          # remove DC offset
    # trim silence so duration reflects the actual call
    trimmed, _ = librosa.effects.trim(wav, top_db=30)
    if trimmed.size < sr * 0.01:
        trimmed = wav
    duration = len(trimmed) / sr

    n_fft = min(C.N_FFT, len(trimmed)) if len(trimmed) >= 256 else 256
    hop = n_fft // 4
    S = np.abs(librosa.stft(trimmed, n_fft=n_fft, hop_length=hop)) + 1e-10
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Separate the call (loud frames) from the background (quiet frames).
    # Marine recordings carry heavy, often non-stationary low-frequency ocean
    # noise, so we work from the loudest frames and subtract the background
    # spectrum estimated from the quiet frames.
    frame_energy = S.sum(axis=0)
    loud = frame_energy >= np.percentile(frame_energy, 75)
    quiet = frame_energy <= np.percentile(frame_energy, 35)
    bg = (S[:, quiet].mean(axis=1, keepdims=True) if quiet.any()
          else np.median(S, axis=1, keepdims=True))

    Sl = S[:, loud] if loud.any() else S            # raw call frames
    # tonality is measured on the *raw* call frames — background subtraction
    # would make every spectrum look artificially tonal.
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=Sl)))

    Scl = np.maximum(Sl - bg, 1e-10)                # noise-cleaned call frames
    Scl[freqs < C.F_MIN, :] = 1e-10                 # drop DC / rumble bins
    call_spec = Scl.mean(axis=1)
    dominant = float(freqs[int(np.argmax(call_spec))])
    centroid = float(np.mean(librosa.feature.spectral_centroid(S=Scl, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=Scl, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(trimmed)))
    lo, hi = _energy_edges(Scl, freqs)

    f0_mean, f0_range, voiced = _pitch(trimmed, sr)
    pulse = _pulse_rate(trimmed, sr)

    return AcousticFeatures(
        duration_s=round(duration, 4),
        dominant_freq_hz=round(dominant, 1),
        spectral_centroid_hz=round(centroid, 1),
        bandwidth_hz=round(bandwidth, 1),
        low_freq_hz=round(lo, 1),
        high_freq_hz=round(hi, 1),
        spectral_flatness=round(flatness, 4),
        zero_crossing_rate=round(zcr, 4),
        f0_mean_hz=round(f0_mean, 1),
        f0_range_hz=round(f0_range, 1),
        voiced_fraction=round(voiced, 3),
        pulse_rate_hz=round(pulse, 1),
    )


def _pitch(wav: np.ndarray, sr: int) -> tuple[float, float, float]:
    """Mean f0, f0 range, and voiced fraction via probabilistic YIN."""
    try:
        f0, voiced_flag, _ = librosa.pyin(
            wav, fmin=_F0_MIN, fmax=_F0_MAX, sr=sr,
            frame_length=min(2048, max(256, len(wav))),
        )
    except Exception:
        return 0.0, 0.0, 0.0
    voiced = f0[~np.isnan(f0)]
    if voiced.size == 0:
        return 0.0, 0.0, 0.0
    frac = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0
    return float(np.mean(voiced)), float(np.ptp(voiced)), frac
