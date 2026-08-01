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
import logging
import time
from contextlib import contextmanager
from functools import lru_cache

import librosa
import numpy as np
import torch

from . import config as C
from .audio import logmel
from .calltype import classify
from .features import extract
from .model import SpeciesCNN
from .quality import assess
from .taxonomy import group_of

log = logging.getLogger(__name__)


@contextmanager
def _timed(stage: str, into: dict):
    """Record how long a stage took, so slow requests can be attributed to one.

    Worth having in the log: the first prediction in a fresh process is far
    slower than the rest (JIT + checkpoint load), and this says which stage
    absorbed it.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        into[stage] = round(time.perf_counter() - start, 3)


@lru_cache(maxsize=1)
def _load_model():
    if not C.CHECKPOINT.exists():
        # Same arity as the loaded case, so callers can unpack either result.
        return None, None, None, None
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    ckpt = torch.load(C.CHECKPOINT, map_location=dev)
    model = SpeciesCNN(ckpt["n_classes"]).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    labels = json.loads(C.LABELMAP.read_text())
    norm = json.loads(C.NORM_STATS.read_text())
    return model, labels, norm, dev


def _group_from(probs: np.ndarray, labels: dict, top_index: int) -> dict | None:
    """Broad group for the top species, with that group's total probability.

    Taken from the top species rather than from an independent argmax so the two
    can never contradict each other on screen. The confidence is the summed mass
    of every species in the group, i.e. the model's belief that it heard *this
    kind* of animal — a claim that holds up far better than the species itself
    (~0.98 vs ~0.79 on the test split).
    """
    label = group_of(labels[str(int(top_index))])
    if label is None:
        return None
    mass = sum(
        float(p) for i, p in enumerate(probs)
        if group_of(labels[str(i)]) == label
    )
    return {"label": label, "confidence": round(mass, 4)}


def _species_topk(wav: np.ndarray, k: int = 3):
    """Top-k species and the broad group, or None when no model is loaded."""
    loaded = _load_model()
    if loaded[0] is None:
        return None, None
    model, labels, norm, dev = loaded
    mel = (logmel(wav) - norm["mean"]) / norm["std"]
    x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float().to(dev)
    with torch.no_grad():
        probs = torch.softmax(model(x), 1).cpu().numpy()[0]
    idx = np.argsort(probs)[::-1][:k]
    topk = [{"species": labels[str(int(i))], "confidence": round(float(probs[i]), 4)}
            for i in idx]
    return topk, _group_from(probs, labels, idx[0])


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
    # Decode at most MAX_AUDIO_SECONDS. Anything under that (every normal clip)
    # is loaded whole and unchanged; a pathologically long file is truncated to
    # its first 10 minutes rather than being allowed to decode into gigabytes.
    # The loudest-window search then runs over whatever was decoded.
    timings: dict[str, float] = {}
    with _timed("decode", timings):
        wav, _ = librosa.load(
            audio_path, sr=C.SAMPLE_RATE, mono=True, duration=C.MAX_AUDIO_SECONDS
        )
    if wav.size == 0:
        raise ValueError("empty audio")

    with _timed("features", timings):
        feats = extract(wav)
    call = classify(feats)
    with _timed("species", timings):
        species, group = _species_topk(wav)

    # A closed-set softmax names a species for silence too, and confidently, so
    # the input is judged on its own terms. Species is still returned for
    # completeness; consumers should treat it as unreliable when this is set.
    warning = assess(wav, feats)

    result = {
        "species": species,                          # None if no trained model yet
        "group": group,                              # broad, far more reliable
        "signal_warning": warning,                   # set when there is no call to identify
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
        with _timed("spectrogram", timings):
            result["spectrogram"] = _spectrogram_png(wav)

    log.info(
        "predict %.1fs audio in %.1fs %s",
        len(wav) / C.SAMPLE_RATE, sum(timings.values()), timings,
    )
    return result


if __name__ == "__main__":
    import sys
    print(json.dumps(predict(sys.argv[1], with_spectrogram=False), indent=2))
