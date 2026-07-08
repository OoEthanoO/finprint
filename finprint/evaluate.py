"""Evaluate the trained species CNN on the held-out WMMS test split.

    python -m finprint.evaluate

Writes a text classification report and a confusion-matrix PNG into reports/.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, top_k_accuracy_score)

from . import config as C
from .model import SpeciesCNN
from .train import device


def load_model(dev):
    ckpt = torch.load(C.CHECKPOINT, map_location=dev)
    model = SpeciesCNN(ckpt["n_classes"]).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["n_classes"]


def main() -> None:
    dev = device()
    labels = json.loads(C.LABELMAP.read_text())
    names = [labels[str(i)] for i in range(len(labels))]
    norm = json.loads(C.NORM_STATS.read_text())

    model, n_classes = load_model(dev)

    d = np.load(C.CACHE_DIR / "test.npz")
    X = ((d["X"] - norm["mean"]) / norm["std"]).astype(np.float32)
    y = d["y"]

    probs = []
    with torch.no_grad():
        for i in range(0, len(X), C.BATCH_SIZE):
            xb = torch.from_numpy(X[i:i + C.BATCH_SIZE]).unsqueeze(1).to(dev)
            probs.append(torch.softmax(model(xb), 1).cpu().numpy())
    probs = np.concatenate(probs)
    pred = probs.argmax(1)

    acc = accuracy_score(y, pred)
    macro_f1 = f1_score(y, pred, average="macro", zero_division=0)
    top3 = top_k_accuracy_score(y, probs, k=3, labels=list(range(n_classes)))

    print(f"test clips : {len(y)}")
    print(f"accuracy   : {acc:.4f}")
    print(f"top-3 acc  : {top3:.4f}")
    print(f"macro-F1   : {macro_f1:.4f}")

    report = classification_report(
        y, pred, labels=list(range(n_classes)), target_names=names, zero_division=0
    )
    (C.REPORTS_DIR / "classification_report.txt").write_text(
        f"accuracy {acc:.4f}  top3 {top3:.4f}  macro-F1 {macro_f1:.4f}\n\n{report}"
    )
    print("\n" + report)

    _plot_confusion(y, pred, names)
    _write_metrics_json(acc, top3, macro_f1, len(y))


def _plot_confusion(y, pred, names) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y, pred, labels=list(range(len(names))), normalize="true")
    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(cm, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Normalized confusion matrix — WMMS test split")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = C.REPORTS_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    print(f"\nconfusion matrix -> {out}")


def _write_metrics_json(acc, top3, macro_f1, n) -> None:
    (C.REPORTS_DIR / "metrics.json").write_text(json.dumps(
        {"accuracy": acc, "top3_accuracy": top3,
         "macro_f1": macro_f1, "n_test": int(n)}, indent=2))


if __name__ == "__main__":
    main()
