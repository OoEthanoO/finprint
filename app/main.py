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

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finprint import config as C  # noqa: E402
from finprint.predict import predict  # noqa: E402

app = FastAPI(title="finprint", description="Marine-mammal call classifier")

STATIC = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")

ALLOWED_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".webm", ".aiff", ".aif"}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model_available": C.CHECKPOINT.exists()}


@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    suffix = Path(file.filename or "clip.wav").suffix.lower() or ".wav"
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"unsupported file type '{suffix}'")

    data = await file.read()
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
