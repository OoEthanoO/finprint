"""Structural call-type classification from acoustic features.

The WMMS data has no behavioral/context labels, so we do NOT claim to detect
"mating" or "hunting". Instead we classify a call by its *acoustic structure*
into the categories bioacousticians define by ear/spectrogram:

  - Echolocation click   : short, broadband, noisy, often in trains
  - Burst-pulse call     : rapid pulse train, partially tonal
  - Whistle              : tonal, narrowband, clear pitch contour
  - Song / moan          : low-frequency, longer, tonal (baleen whales)
  - Broadband pulse/grunt: fallback for everything else

Each rule is a transparent score over the measured features. Thresholds are
approximate and grounded in general marine-mammal bioacoustics; the goal is an
explainable second output, not a second trained model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import AcousticFeatures

CATEGORIES = [
    "Echolocation click",
    "Burst-pulse call",
    "Whistle",
    "Song / moan",
    "Broadband pulse / grunt",
]


@dataclass
class CallType:
    label: str
    confidence: float
    rationale: str
    scores: dict          # category -> normalized score


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def classify(f: AcousticFeatures) -> CallType:
    # Soft feature gates in [0, 1]. Each category multiplies the gates that
    # *define* it, so a category can only win when its key traits are present
    # (an averaged score would let, say, "song" win for a 10 kHz dolphin).
    noisy = _clamp01(f.spectral_flatness / 0.15)          # broadband / click-like
    tonal = 1.0 - noisy
    high_freq = _clamp01((f.dominant_freq_hz - 1200.0) / 2000.0)
    low_freq = _clamp01((1500.0 - f.dominant_freq_hz) / 1500.0)
    voiced_g = _clamp01(f.voiced_fraction / 0.30)
    long_g = _clamp01((f.duration_s - 0.8) / 1.5)
    bright = _clamp01(f.zero_crossing_rate / 0.30)         # high-frequency proxy
    pulsed = _clamp01(1.0 - abs(np.log10(max(f.pulse_rate_hz, 1e-6) / 150.0)) / 1.2) \
        if f.pulse_rate_hz > 0 else 0.0

    scores = {
        # broadband + noisy, and either high-frequency or high zero-crossing
        "Echolocation click": noisy * max(high_freq, bright),
        # a genuine pulse train, still partly tonal
        "Burst-pulse call": pulsed * (0.4 + 0.6 * tonal),
        # tonal + clear pitch + mid/high frequency, not pulsed
        "Whistle": tonal * voiced_g * high_freq * (1.0 - 0.5 * (pulsed > 0)),
        # tonal + low frequency + sustained
        "Song / moan": tonal * low_freq * (0.4 + 0.6 * long_g),
        # noisy + low frequency + little pitch (fallback grunt/knock)
        "Broadband pulse / grunt": noisy * low_freq * _clamp01(1.0 - voiced_g),
    }

    # normalize to a pseudo-probability distribution, sharpened so the winning
    # category is decisive rather than a near-tie
    raw = np.array([scores[c] for c in CATEGORIES], dtype=np.float64)
    raw = np.maximum(raw, 1e-4) ** 2.0
    probs = raw / raw.sum()
    order = np.argsort(probs)[::-1]
    best = CATEGORIES[order[0]]
    conf = float(probs[order[0]])

    norm_scores = {CATEGORIES[i]: round(float(probs[i]), 3) for i in order}
    return CallType(
        label=best,
        confidence=round(conf, 3),
        rationale=_rationale(best, f),
        scores=norm_scores,
    )


def _rationale(label: str, f: AcousticFeatures) -> str:
    if label == "Echolocation click":
        return (f"broadband (bw {f.bandwidth_hz:.0f} Hz), noise-like "
                f"(flatness {f.spectral_flatness:.2f}) and high-frequency "
                f"(dominant {f.dominant_freq_hz:.0f} Hz)")
    if label == "Burst-pulse call":
        return (f"repetitive pulse train (~{f.pulse_rate_hz:.0f} pulses/s) "
                f"with partial tonal structure")
    if label == "Whistle":
        return (f"tonal (flatness {f.spectral_flatness:.2f}), narrowband, "
                f"clear pitch (f0 ~{f.f0_mean_hz:.0f} Hz, "
                f"voiced {f.voiced_fraction*100:.0f}%)")
    if label == "Song / moan":
        return (f"low-frequency (dominant {f.dominant_freq_hz:.0f} Hz), "
                f"sustained ({f.duration_s:.2f} s) and tonal")
    return (f"broadband ({f.bandwidth_hz:.0f} Hz) without a clear pitch or "
            f"pulse structure")
