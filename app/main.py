"""finprint web app — FastAPI backend.

    uvicorn app.main:app --reload

GET  /              -> single-page UI
POST /api/predict   -> multipart audio file -> JSON prediction
GET  /api/health    -> readiness + whether a trained model is loaded
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finprint import config as C  # noqa: E402
from finprint.predict import predict  # noqa: E402

app = FastAPI(title="finprint", description="Marine-mammal call classifier")

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
