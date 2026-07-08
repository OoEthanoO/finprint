"""End-to-end inference: an audio file -> a structured prediction.

    from finprint.predict import predict
    predict("clip.wav")  # -> dict with species, call type, features, spectrogram

Model/label artifacts are loaded once and cached. The acoustic + call-type parts
work even without a trained checkpoint, so the pipeline degrades gracefully.
"""

from __future__ import annotations

import base64
import io
import json
from functools import lru_cache

import librosa
import numpy as np
import torch

from . import config as C
from .audio import logmel
from .calltype import classify
from .features import extract
from .model import SpeciesCNN


@lru_cache(maxsize=1)
def _load_model():
    if not C.CHECKPOINT.exists():
        return None, None, None
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    ckpt = torch.load(C.CHECKPOINT, map_location=dev)
    model = SpeciesCNN(ckpt["n_classes"]).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    labels = json.loads(C.LABELMAP.read_text())
    norm = json.loads(C.NORM_STATS.read_text())
    return model, labels, norm, dev


def _species_topk(wav: np.ndarray, k: int = 3):
    loaded = _load_model()
    if loaded[0] is None:
        return None
    model, labels, norm, dev = loaded
    mel = (logmel(wav) - norm["mean"]) / norm["std"]
    x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float().to(dev)
    with torch.no_grad():
        probs = torch.softmax(model(x), 1).cpu().numpy()[0]
    idx = np.argsort(probs)[::-1][:k]
    return [{"species": labels[str(int(i))], "confidence": round(float(probs[i]), 4)}
            for i in idx]


def _spectrogram_png(wav: np.ndarray) -> str:
    """Render a log-mel spectrogram as a base64 PNG for the web UI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mel = logmel(wav)
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.imshow(mel, origin="lower", aspect="auto", cmap="magma")
    ax.set_xlabel("time frames")
    ax.set_ylabel("mel bins")
    ax.set_title("log-mel spectrogram")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def predict(audio_path: str, with_spectrogram: bool = True) -> dict:
    wav, _ = librosa.load(audio_path, sr=C.SAMPLE_RATE, mono=True)
    if wav.size == 0:
        raise ValueError("empty audio")

    feats = extract(wav)
    call = classify(feats)
    species = _species_topk(wav)

    result = {
        "species": species,                          # None if no trained model yet
        "call_type": {
            "label": call.label,
            "confidence": call.confidence,
            "rationale": call.rationale,
            "scores": call.scores,
        },
        "features": feats.as_dict(),
        "model_available": species is not None,
    }
    if with_spectrogram:
        result["spectrogram"] = _spectrogram_png(wav)
    return result


if __name__ == "__main__":
    import sys
    print(json.dumps(predict(sys.argv[1], with_spectrogram=False), indent=2))
