"""Central configuration for finprint.

Every knob that the data pipeline, the model, and inference all need to agree
on lives here, so training and serving can never drift out of sync.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

for _d in (DATA_DIR, CACHE_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- dataset -------------------------------------------------------------
HF_DATASET = "confit/wmms-parquet"

# --- audio / spectrogram -------------------------------------------------
# WMMS clips are wildly heterogeneous: native sample rates 600 Hz - 166 kHz,
# durations 0.2 s - 114 s. We resample everything to a single rate and take a
# fixed-length, highest-energy window so spectrograms are batchable and the
# window lands on actual call content rather than silence.
SAMPLE_RATE = 32000          # Nyquist 16 kHz — keeps dolphin whistle energy
CLIP_SECONDS = 4.0           # covers ~80% of clips fully; longer are windowed
N_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
F_MIN = 20.0
F_MAX = SAMPLE_RATE / 2

# Coarse stride (samples) for the energy-based window search.
ENERGY_WIN_STRIDE = SAMPLE_RATE // 4

# The DSP layer measures the same-size highest-energy window the CNN sees, so
# the printed numbers describe the call rather than two minutes of ocean, and so
# pitch tracking (cost grows with length) stays bounded on a 114 s clip.
ANALYSIS_SECONDS = CLIP_SECONDS
ANALYSIS_SAMPLES = int(SAMPLE_RATE * ANALYSIS_SECONDS)

# Number of mel frames in one clip (used to size the model / augmentation).
N_FRAMES = N_SAMPLES // HOP_LENGTH + 1

# --- serving limits ------------------------------------------------------
# The public demo runs on a single small scale-to-zero instance, so one large
# upload must not be able to exhaust its memory or monopolize the worker.
# librosa decodes audio into a float32 array (4 bytes/sample), i.e. ~128 KB per
# second at SAMPLE_RATE, so these two caps bound both the bytes on the wire and
# the size of the decoded array.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024      # reject the request above this (~25 MB)
MAX_AUDIO_SECONDS = 600.0                # decode at most the first 10 minutes

# --- confidence reporting ------------------------------------------------
# Below this top-species probability the app labels a prediction "low
# confidence" rather than presenting it as an answer. Measured, not chosen: on
# the held-out split, predictions at or above this line are ~0.97 correct and
# those below it ~0.49. It lives here because three places must agree on it —
# `predict()` sets the flag, the UI renders it, and `finprint.evaluate`
# re-measures the split behind it on every retrain.
LOW_CONFIDENCE = 0.5

# --- training ------------------------------------------------------------
SEED = 1234
BATCH_SIZE = 32
EPOCHS = 60
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.15          # carved out of the train split for early stopping
EARLY_STOP_PATIENCE = 12

# --- build identity ------------------------------------------------------
# Which commit is actually serving. Set at deploy time, because the running
# image cannot work this out for itself: `.gcloudignore` strips `.git` from the
# build context, so there is no repository inside the container to ask.
#
#   gcloud run deploy ... --update-env-vars FINPRINT_GIT_SHA=$(git rev-parse HEAD)
#
# "unknown" is the honest answer for a local run or a deploy that forgot the
# flag — never a stale value, which would be worse than no value at all.
GIT_SHA = os.environ.get("FINPRINT_GIT_SHA", "unknown")

# --- artifacts -----------------------------------------------------------
CHECKPOINT = MODELS_DIR / "species_cnn.pt"
LABELMAP = MODELS_DIR / "labels.json"
NORM_STATS = MODELS_DIR / "norm.json"
