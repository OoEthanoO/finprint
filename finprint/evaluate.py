"""Evaluate the trained species CNN on the held-out WMMS test split.

    python -m finprint.evaluate

Writes a text classification report and a confusion-matrix PNG into reports/.

Species accuracy is not the number the app leads with. The UI shows the broad
group first and hides the species behind a confidence line, so both of those are
measured here too — otherwise the figures quoted in the README would be
hand-measured once and left to rot through the next retrain.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, top_k_accuracy_score)

from . import config as C
from .model import SpeciesCNN
from .taxonomy import group_confidence, group_of
from .train import device


def load_model(dev):
    ckpt = torch.load(C.CHECKPOINT, map_location=dev)
    model = SpeciesCNN(ckpt["n_classes"]).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["n_classes"]


# Bin edges for the group-confidence calibration table. The first two bins are
# what the app treats as weak evidence; the split at 0.8 is where reported
# confidence historically starts being near-perfect, and keeping the boundary
# fixed is what makes the table comparable between retrains.
_CALIBRATION_BINS = ((0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.0 + 1e-9))


def group_metrics(probs: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    """How well the coarse group label — the app's headline — actually does.

    The group shown to a user is the *top species'* group (never an independent
    argmax, so the two lines on screen cannot contradict each other), and the
    confidence beside it is that group's summed probability mass. Both are
    measured here exactly as `finprint.predict` computes them.

    Clips whose true species has no group are excluded rather than counted
    wrong; `tests/test_predict.py` separately asserts that set is empty.
    """
    pred = probs.argmax(1)
    true_group = [group_of(names[i]) for i in y]
    pred_group = [group_of(names[i]) for i in pred]

    kept = [i for i, g in enumerate(true_group) if g is not None]
    correct = [true_group[i] == pred_group[i] for i in kept]
    accuracy = float(np.mean(correct)) if kept else 0.0

    # Reported confidence vs. what it turned out to be worth.
    reported = [
        group_confidence(probs[i], names, pred_group[i])
        if pred_group[i] is not None else 0.0
        for i in kept
    ]
    calibration = []
    for lo, hi in _CALIBRATION_BINS:
        in_bin = [j for j, c in enumerate(reported) if lo <= c < hi]
        if not in_bin:
            continue
        calibration.append({
            "confidence_range": [lo, round(min(hi, 1.0), 2)],
            "n": len(in_bin),
            "accuracy": round(float(np.mean([correct[j] for j in in_bin])), 4),
        })

    return {
        "accuracy": round(accuracy, 4),
        "n": len(kept),
        "calibration": calibration,
    }


def confidence_metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    """What the low-confidence line is worth on this split.

    The app hides a species behind `C.LOW_CONFIDENCE`, so the claim that matters
    is not overall accuracy but accuracy *among the predictions it presents as
    answers*. Reported alongside the flagged share, because a threshold that
    flags everything would score perfectly on the first number alone.

    Both counts come back with the accuracies, and that is not decoration: under
    heavy degradation the app flags nearly everything, so "accuracy when
    confident" can be computed from one or two clips and swing between 0.0 and
    1.0 on a single prediction. The reader needs the denominator to know which
    numbers to ignore.
    """
    pred = probs.argmax(1)
    top = probs.max(1)
    confident = top >= C.LOW_CONFIDENCE
    hit = pred == y

    def _acc(mask):
        return round(float(hit[mask].mean()), 4) if mask.any() else None

    return {
        "threshold": C.LOW_CONFIDENCE,
        "flagged_fraction": round(float((~confident).mean()), 4),
        "n_confident": int(confident.sum()),
        "n_flagged": int((~confident).sum()),
        "accuracy_when_confident": _acc(confident),
        "accuracy_when_flagged": _acc(~confident),
    }


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

    groups = group_metrics(probs, y, names)
    conf = confidence_metrics(probs, y)

    print(f"test clips : {len(y)}")
    print(f"accuracy   : {acc:.4f}")
    print(f"top-3 acc  : {top3:.4f}")
    print(f"macro-F1   : {macro_f1:.4f}")
    print(f"group acc  : {groups['accuracy']:.4f}   (the label the app leads with)")
    print(
        f"\nlow-confidence line at {conf['threshold']}: "
        f"{conf['flagged_fraction']:.1%} of predictions flagged; "
        f"accuracy {conf['accuracy_when_confident']} when confident "
        f"vs {conf['accuracy_when_flagged']} when flagged"
    )
    if groups["calibration"]:
        print("\ngroup confidence   n   actual group accuracy")
        for b in groups["calibration"]:
            lo, hi = b["confidence_range"]
            print(f"  {lo:.2f} - {hi:.2f}   {b['n']:>4}   {b['accuracy']:.4f}")

    report = classification_report(
        y, pred, labels=list(range(n_classes)), target_names=names, zero_division=0
    )
    (C.REPORTS_DIR / "classification_report.txt").write_text(
        f"accuracy {acc:.4f}  top3 {top3:.4f}  macro-F1 {macro_f1:.4f}\n\n{report}"
    )
    print("\n" + report)

    _plot_confusion(y, pred, names)
    _write_metrics_json(acc, top3, macro_f1, len(y), groups, conf)


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


def _write_metrics_json(acc, top3, macro_f1, n, groups, conf) -> None:
    (C.REPORTS_DIR / "metrics.json").write_text(json.dumps(
        {"accuracy": acc, "top3_accuracy": top3,
         "macro_f1": macro_f1, "n_test": int(n),
         "group": groups, "low_confidence": conf}, indent=2))


if __name__ == "__main__":
    main()
