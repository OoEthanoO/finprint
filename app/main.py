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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

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

# Allow a separately-hosted frontend (e.g. on Vercel) to call this API. This is
# a public, read-only inference endpoint, so "*" is fine; restrict to your own
# domain(s) via the FINPRINT_ALLOW_ORIGINS env var if you prefer.
_origins = os.environ.get("FINPRINT_ALLOW_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# Everything ffmpeg can hand to librosa. The gate is on the extension only —
# content is sniffed at decode time, so a mislabelled file still works (Safari's
# recorder emits MP4/AAC while the page names the blob .webm, and that decodes
# identically). .mp4 is here because a phone recording or a downloaded clip is
# often a video container; the audio track is extracted, and a file with no
# audio track fails the decode with a clear 422 rather than being turned away
# at the door.
ALLOWED_SUFFIXES = {
    ".wav", ".flac", ".ogg", ".mp3", ".m4a", ".webm", ".aiff", ".aif",
    ".mp4", ".aac", ".opus",
}


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
    # Without this, browsers derive their own freshness window from
    # Last-Modified and can keep serving a stale page well after a deploy — a
    # returning visitor sees the old UI with no obvious way to know.
    #
    # `no-cache` means "revalidate before reusing". Starlette's FileResponse
    # sets an ETag but implements no conditional handling (it never returns
    # 304), so each revalidation re-sends the page rather than a bodiless 304.
    # At ~13 KB that is a cheap price for a deploy always being visible.
    return FileResponse(
        STATIC / "index.html", headers={"Cache-Control": "no-cache"}
    )


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
            # predict() is ~0.7 s of blocking CPU work (mostly pitch tracking).
            # Calling it directly from this coroutine parks the event loop for
            # that whole time, and the loop is what serves everything else — a
            # page load during a single upload measured 1.84 s instead of 1 ms.
            # A worker thread keeps the loop free; the temp file stays alive
            # because we await inside its `with` block.
            result = await run_in_threadpool(predict, tmp.name)
        except Exception as exc:  # decode / inference failure
            raise HTTPException(422, f"could not process audio: {exc}") from exc

    return JSONResponse(result)
