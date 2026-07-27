"""Build-time warm-up: bake the matplotlib/numba caches into the image.

    python -m scripts.warmup

The real work lives in :mod:`finprint.warmup`, which the app also runs at
startup. Running it here as well populates ``MPLCONFIGDIR`` / ``NUMBA_CACHE_DIR``
inside the image and doubles as a smoke test that the built image can predict.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finprint.warmup import run  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    print(f"warmup ok in {run():.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
