"""Measure what survives degraded audio, and regenerate the robustness tables.

    python -m scripts.robustness              # 90 clips, the published setting
    python -m scripts.robustness --n 340      # the whole test split
    python -m scripts.robustness --list       # conditions, without running them

Writes reports/robustness.json and prints the two markdown tables that live in
reports/error_analysis.md.

Everything the model has seen is a clean Watkins archive recording, and a demo
often is not — so the interesting question is not "how accurate is it" but "which
of these claims still holds when someone plays a call through a laptop speaker".
Answering it by hand once meant the answer would quietly describe an older model
after the next retrain, which is the same reason `finprint.evaluate` grew the
group and confidence metrics it reuses here.

Reuses those definitions rather than restating them: the group accuracy in a
degraded row is computed by the same code as the headline group accuracy, so the
rows are comparable to the clean baseline and to each other.

Needs the raw test audio (not the cached spectrograms — the whole point is to
degrade the waveform), so this re-reads the dataset. ffmpeg must be on PATH for
the mp3 rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from finprint import config as C  # noqa: E402
from finprint.audio import logmel  # noqa: E402
from finprint.degrade import CONDITIONS  # noqa: E402
from finprint.evaluate import confidence_metrics, group_metrics  # noqa: E402
from finprint.predict import _load_model  # noqa: E402


def load_clips(n: int, seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    """A reproducible sample of decoded test-split waveforms and their labels."""
    from datasets import Audio, load_dataset

    from scripts.prepare_data import decode_audio

    ds = load_dataset(C.HF_DATASET, split="test").cast_column("audio", Audio(decode=False))
    idx = np.arange(len(ds))
    if n < len(ds):
        # A fixed seed, so re-running quotes the same clips rather than a fresh
        # sample that would move every number a little for no reason.
        idx = np.random.default_rng(seed).choice(len(ds), size=n, replace=False)
        idx.sort()

    wavs, y = [], []
    for i in idx:
        ex = ds[int(i)]
        wavs.append(decode_audio(ex["audio"]))
        y.append(int(ex["label"]))
    return wavs, np.asarray(y, dtype=np.int64)


def score(wavs, model, norm, dev) -> np.ndarray:
    """Softmax probabilities for a list of waveforms, the way serving does it."""
    out = []
    for i in range(0, len(wavs), C.BATCH_SIZE):
        mels = [(logmel(w) - norm["mean"]) / norm["std"] for w in wavs[i:i + C.BATCH_SIZE]]
        x = torch.from_numpy(np.stack(mels)).unsqueeze(1).float().to(dev)
        with torch.no_grad():
            out.append(torch.softmax(model(x), 1).cpu().numpy())
    return np.concatenate(out)


def run(n: int, seed: int) -> dict:
    model, labels, norm, dev = _load_model()
    if model is None:
        raise SystemExit(
            f"no trained checkpoint at {C.CHECKPOINT} — run `python -m finprint.train`"
        )
    names = [labels[str(i)] for i in range(len(labels))]

    print(f"loading {n} test clips ...", flush=True)
    wavs, y = load_clips(n, seed)
    print(f"  {len(wavs)} clips, {len(set(y.tolist()))} species\n", flush=True)

    rows = {}
    for name, fn in CONDITIONS.items():
        # Each condition gets its own generator seeded from the run seed, so a
        # row is reproducible on its own and adding a condition does not shift
        # the noise in the rows after it. crc32 rather than hash(): Python
        # randomizes string hashing per process, which would quietly make these
        # numbers unreproducible between runs.
        rng = np.random.default_rng(zlib.crc32(name.encode()) ^ seed)
        print(f"  {name} ...", flush=True)
        probs = score([fn(w, rng) for w in wavs], model, norm, dev)

        species_acc = float((probs.argmax(1) == y).mean())
        rows[name] = {
            "species_accuracy": round(species_acc, 4),
            "group_accuracy": group_metrics(probs, y, names)["accuracy"],
            "mean_confidence": round(float(probs.max(1).mean()), 4),
            **confidence_metrics(probs, y),
        }
    return {"n_clips": len(wavs), "seed": seed, "conditions": rows}


def print_tables(report: dict) -> None:
    rows = report["conditions"]
    print(f"\n### Degraded audio — {report['n_clips']} held-out clips\n")
    print("| Condition | Species | Group | Mean confidence |")
    print("|---|---|---|---|")
    for name, r in rows.items():
        print(f"| {name} | {r['species_accuracy']:.3f} | "
              f"{r['group_accuracy']:.3f} | {r['mean_confidence']:.2f} |")

    print("\n### Does the app know when it is struggling?\n")
    print("| Condition | % flagged low-confidence | Clips still shown | "
          "Accuracy when confident |")
    print("|---|---|---|---|")
    for name, r in rows.items():
        conf = r["accuracy_when_confident"]
        # n is printed next to the accuracy because the bottom rows compute it
        # from a handful of clips, where it means very little.
        shown = "—" if conf is None else f"{conf:.3f}"
        print(f"| {name} | {r['flagged_fraction']:.0%} | {r['n_confident']} | {shown} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=90, help="clips to sample (default 90)")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--list", action="store_true", help="list conditions and exit")
    args = ap.parse_args()

    if args.list:
        for name in CONDITIONS:
            print(name)
        return

    report = run(args.n, args.seed)
    out = C.REPORTS_DIR / "robustness.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print_tables(report)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
