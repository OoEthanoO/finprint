"""Deterministic synthetic audio for the tests.

Real WMMS clips are heavy and non-deterministic to fetch, so the tests drive the
DSP with signals whose acoustic properties we control exactly: a pure tone is
tonal and unpulsed, white noise is broadband and unvoiced, an on/off carrier is
a pulse train at a known rate.
"""
from __future__ import annotations

import numpy as np

import finprint.config as C

SR = C.SAMPLE_RATE


def tone(freq: float, seconds: float, amp: float = 0.6,
         noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """A sine tone, optionally with a little additive noise."""
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    t = np.arange(n) / SR
    w = amp * np.sin(2 * np.pi * freq * t)
    if noise:
        w = w + noise * rng.standard_normal(n)
    return w.astype(np.float32)


def white_noise(seconds: float, amp: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(SR * seconds))).astype(np.float32)


def pulse_train(carrier: float, rate_hz: float, seconds: float,
                amp: float = 0.6) -> np.ndarray:
    """A carrier gated on/off at `rate_hz` — an envelope pulse train."""
    n = int(SR * seconds)
    t = np.arange(n) / SR
    gate = (np.sin(2 * np.pi * rate_hz * t) >= 0).astype(np.float32)
    return (amp * gate * np.sin(2 * np.pi * carrier * t)).astype(np.float32)
