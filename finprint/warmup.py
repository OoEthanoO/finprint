"""Force the one-time work a fresh process would otherwise do inside a user's
first request.

A cold process spends ~50 s on initialisation that has nothing to do with the
uploaded clip: librosa imports its submodules lazily on first use, numba
JIT-compiles the pitch kernels, matplotlib builds a font cache, and torch loads
the checkpoint. Running one throwaway prediction pays all of it up front.

Called from two places:

* the Dockerfile, at image build time -- bakes the matplotlib/numba disk caches
  into the image and smoke-tests that the image can actually predict;
* the app's startup (``app.main``) -- so the JIT runs while Cloud Run's startup
  CPU boost is still active and before the port opens, rather than inside the
  first request at normal CPU.
"""

from __future__ import annotations

import logging
import tempfile
import time

import numpy as np
import soundfile as sf

from . import config as C

log = logging.getLogger(__name__)


def _synthetic_clip() -> np.ndarray:
    """A short frequency-modulated tone: voiced enough to drive the pitch
    tracker (the expensive numba path) without making startup slow."""
    t = np.arange(int(C.SAMPLE_RATE * 1.5)) / C.SAMPLE_RATE
    f0 = 800.0 + 200.0 * np.sin(2 * np.pi * 2.0 * t)
    phase = 2 * np.pi * np.cumsum(f0) / C.SAMPLE_RATE
    return (0.5 * np.sin(phase)).astype(np.float32)


def run() -> float:
    """Run one throwaway prediction. Returns how long it took, in seconds.

    Never raises: a failed warm-up should cost the first request its latency,
    not stop the service from starting.
    """
    # Imported here rather than at module scope so that importing this module is
    # itself cheap, and so a broken checkpoint cannot break startup.
    from .predict import predict

    started = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, _synthetic_clip(), C.SAMPLE_RATE)
            predict(tmp.name, with_spectrogram=True)
    except Exception:
        log.exception("warm-up failed; first request will pay the cost instead")
        return time.perf_counter() - started

    elapsed = time.perf_counter() - started
    log.info("warm-up complete in %.1fs", elapsed)
    return elapsed
