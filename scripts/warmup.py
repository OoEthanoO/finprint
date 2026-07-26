"""Pre-build, at image build time, the caches a cold instance would otherwise
pay for inside a user's first request.

A freshly started container spends ~50 s on work that is identical every time:

  * matplotlib builds its font cache the first time a figure is rendered
    (``MPLCONFIGDIR``);
  * numba JIT-compiles the kernels librosa uses for pitch tracking
    (``NUMBA_CACHE_DIR``);
  * torch loads the checkpoint.

Running one real prediction here writes the first two into the image, so a cold
instance only pays the unavoidable per-process cost (imports + checkpoint load)
rather than recompiling from scratch. Both cache dirs must be set and writable
when this runs -- see the Dockerfile.

    python -m scripts.warmup
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finprint import config as C  # noqa: E402
from finprint.predict import predict  # noqa: E402


def _synthetic_clip() -> np.ndarray:
    """A short frequency-modulated tone: voiced enough to drive the pitch
    tracker (the expensive numba path) without making the build slow."""
    t = np.arange(int(C.SAMPLE_RATE * 1.5)) / C.SAMPLE_RATE
    f0 = 800.0 + 200.0 * np.sin(2 * np.pi * 2.0 * t)
    phase = 2 * np.pi * np.cumsum(f0) / C.SAMPLE_RATE
    return (0.5 * np.sin(phase)).astype(np.float32)


def main() -> int:
    started = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, _synthetic_clip(), C.SAMPLE_RATE)
        result = predict(tmp.name, with_spectrogram=True)

    print(
        f"warmup ok in {time.perf_counter() - started:.1f}s "
        f"(call_type={result['call_type']['label']}, "
        f"model_available={result['model_available']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
