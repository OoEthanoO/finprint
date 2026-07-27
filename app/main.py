"""finprint web app — FastAPI backend.

    uvicorn app.main:app --reload

GET  /              -> single-page UI
POST /api/predict   -> multipart audio file -> JSON prediction
GET  /api/health    -> readiness + whether a trained model is loaded
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# uvicorn only configures its own loggers, leaving the root logger at WARNING —
# so finprint's INFO records (the per-stage prediction timings) would be dropped.
# Cloud Run collects stdout, so a plain stream handler is all that's needed.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)

from finprint import config as C  # noqa: E402
from finprint.predict import predict  # noqa: E402
from finprint.warmup import run as warm_up  # noqa: E402

log = logging.getLogger("finprint.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the process before the port opens.

    uvicorn runs this to completion *before* it binds the socket, so the
    container is not reportable as ready until the first (expensive) prediction
    path has been exercised. On Cloud Run that matters twice over: startup CPU
    boost is still active here but not during an ordinary request, and an
    instance started ahead of traffic absorbs the cost entirely.

    Set FINPRINT_WARMUP=0 to skip it -- worth doing with `--reload`, where the
    cost would be paid on every code change.
    """
    if os.environ.get("FINPRINT_WARMUP", "1") != "0":
        warm_up()
    else:
        log.info("warm-up skipped (FINPRINT_WARMUP=0)")
    yield


app = FastAPI(
    title="finprint",
    description="Marine-mammal call classifier",
    lifespan=lifespan,
)

STATIC = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")

ALLOWED_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".webm", ".aiff", ".aif"}


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    """Reject an over-large request from its declared Content-Length, before the
    body is parsed. This matters on Cloud Run, where the filesystem is in-memory
    tmpfs: Starlette would otherwise spool the whole multipart body there and the
    upload would cost RAM before any handler runs. A client that omits or lies
    about Content-Length still hits the capped read in the endpoint below."""
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > C.MAX_UPLOAD_BYTES:
                mb = C.MAX_UPLOAD_BYTES // (1024 * 1024)
                return JSONResponse(
                    {"detail": f"file too large (limit {mb} MB)"}, status_code=413
                )
        except ValueError:
            pass  # unparseable header — let the capped read handle it
    return await call_next(request)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_available": C.CHECKPOINT.exists(),
        "max_upload_bytes": C.MAX_UPLOAD_BYTES,
        "max_audio_seconds": C.MAX_AUDIO_SECONDS,
    }


async def _read_capped(file: UploadFile, limit: int) -> bytes:
    """Read the upload but stop once it exceeds `limit`, so an oversized file
    never gets fully buffered in memory. Reads one extra byte past the limit to
    distinguish 'exactly at the limit' from 'over'."""
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            413, f"file too large (limit {limit // (1024 * 1024)} MB)"
        )
    return data


@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    suffix = Path(file.filename or "clip.wav").suffix.lower() or ".wav"
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"unsupported file type '{suffix}'")

    data = await _read_capped(file, C.MAX_UPLOAD_BYTES)
    if not data:
        raise HTTPException(400, "empty upload")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            result = predict(tmp.name)
        except Exception as exc:  # decode / inference failure
            raise HTTPException(422, f"could not process audio: {exc}") from exc

    return JSONResponse(result)
