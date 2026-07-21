"""HTTP contract for the FastAPI app, including the upload guard.

The guard has two layers: a Content-Length middleware that rejects before the
body is parsed, and a capped read in the endpoint for clients that omit or lie
about Content-Length. Both are exercised here.
"""
import asyncio
import io
from tempfile import SpooledTemporaryFile

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.datastructures import Headers, UploadFile

import finprint.config as C
from app.main import _read_capped, app

client = TestClient(app)


def _wav_bytes(freq: float = 1000.0, seconds: float = 1.0) -> bytes:
    t = np.arange(int(C.SAMPLE_RATE * seconds)) / C.SAMPLE_RATE
    w = (0.6 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, w, C.SAMPLE_RATE, format="WAV")
    return buf.getvalue()


def _upload(data: bytes) -> UploadFile:
    spool = SpooledTemporaryFile()
    spool.write(data)
    spool.seek(0)
    return UploadFile(file=spool, size=len(data), filename="clip.wav",
                      headers=Headers({"content-type": "audio/wav"}))


def test_health_reports_status_and_limits():
    j = client.get("/api/health").json()
    assert j["status"] == "ok"
    assert isinstance(j["model_available"], bool)
    assert j["max_upload_bytes"] == C.MAX_UPLOAD_BYTES
    assert j["max_audio_seconds"] == C.MAX_AUDIO_SECONDS


def test_predict_rejects_unsupported_type():
    r = client.post("/api/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_predict_rejects_empty_upload():
    r = client.post("/api/predict", files={"file": ("x.wav", b"", "audio/wav")})
    assert r.status_code == 400


def test_predict_returns_prediction_for_valid_audio():
    r = client.post("/api/predict",
                    files={"file": ("clip.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert {"call_type", "features", "spectrogram", "model_available"} <= set(j)
    assert {"label", "confidence", "rationale", "scores"} <= set(j["call_type"])
    assert j["spectrogram"].startswith("data:image/png;base64,")


def test_predict_413_on_oversize_via_middleware(monkeypatch):
    monkeypatch.setattr(C, "MAX_UPLOAD_BYTES", 100)
    r = client.post("/api/predict",
                    files={"file": ("clip.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 413


def test_read_capped_allows_within_limit():
    assert asyncio.run(_read_capped(_upload(b"x" * 50), 100)) == b"x" * 50


def test_read_capped_rejects_over_limit():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(_read_capped(_upload(b"x" * 200), 100))
    assert exc.value.status_code == 413
