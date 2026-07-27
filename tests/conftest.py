"""Make the repo root importable so `finprint` and `app` resolve when pytest is
run from anywhere, and expose the signal helpers to every test module."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# The app warms itself up on startup by running a real prediction (~50 s: librosa
# imports plus numba JIT). Today's tests drive TestClient without the context
# manager, so Starlette never fires the lifespan hook and this costs nothing --
# but that is incidental, and `with TestClient(app)` would silently add ~50 s to
# a run. Opt out explicitly, before app.main is imported.
os.environ.setdefault("FINPRINT_WARMUP", "0")
