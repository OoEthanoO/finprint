# finprint — production image for a container host (Fly.io, Render, Cloud Run, …)
#
# Pinned to Python 3.11 (the version the project targets) so torch/torchaudio
# have wheels — independent of whatever Python the host machine happens to run.
# torch + torchaudio come from the CPU wheel index: no CUDA, ~1.5 GB image
# instead of ~5 GB, and this app does inference on CPU anyway.

FROM python:3.11-slim AS runtime

# Audio decoding: libsndfile for soundfile (wav/flac/ogg), ffmpeg for the
# compressed formats the API accepts (mp3/m4a/webm) via librosa's audioread path.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# Deliberately NOT setting PYTHONDONTWRITEBYTECODE: the usual reason (keeping
# .pyc out of a layer) is the wrong trade here. librosa imports its submodules
# lazily on first use, so without cached bytecode every cold process recompiles
# librosa/numba/scipy from source *inside the first request* -- measured at 33 s
# of a 56 s cold start. `compileall` below bakes the .pyc into the image.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Cache dirs baked into the image and pre-populated by scripts.warmup below.
    # Under /tmp these would be empty on every cold start, and rebuilding them
    # (matplotlib fonts, numba JIT of librosa's pitch kernels) costs ~50 s
    # inside the first request.
    MPLCONFIGDIR=/opt/caches/mpl \
    NUMBA_CACHE_DIR=/opt/caches/numba \
    PORT=8000

WORKDIR /app

# Dependencies first so the layer caches across code changes.
#
# Serving deps only: the training/data-prep stack (datasets -> pyarrow + pandas,
# huggingface_hub, tqdm) is never imported by the API, and pulling it in costs
# ~200 MB of image that every cold start has to fetch. requirements.txt remains
# the full development superset; CI installs this same file, so a serving import
# missing from it fails there rather than in production.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu torch torchaudio \
 && pip install --no-cache-dir -r requirements-serve.txt

# App code (config.py creates data/ models/ reports/ under /app at import,
# so /app must be writable by the runtime user).
COPY . .

# Bake bytecode for the dependencies and the app while still root (site-packages
# is root-owned; appuser could not write these at runtime). Compilation failures
# in unrelated vendored files are not fatal, hence the `|| true`.
RUN python -m compileall -q -j 0 /usr/local/lib/python3.11/site-packages /app || true

RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /opt/caches/mpl /opt/caches/numba \
 && chown -R appuser:appuser /app /opt/caches
USER appuser

# Populate the caches as the runtime user, so they are readable (and the numba
# cache keys match) when the same user serves requests.
RUN python -m scripts.warmup \
 && echo "numba cache entries: $(ls /opt/caches/numba | wc -l)" \
 && echo "mpl cache entries: $(ls /opt/caches/mpl | wc -l)"

EXPOSE 8000

# Hosts inject $PORT; default to 8000 for a plain `docker run`.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
