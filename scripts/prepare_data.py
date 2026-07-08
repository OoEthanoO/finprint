"""Download the WMMS dataset and cache log-mel spectrograms + labels.

Run once:  python -m scripts.prepare_data
Produces:  data/cache/{train,test}.npz  and  models/labels.json
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import Audio, load_dataset  # noqa: E402

from finprint import config as C  # noqa: E402
from finprint.audio import logmel  # noqa: E402


def decode_audio(entry: dict) -> np.ndarray:
    """Decode one raw audio record ({'bytes','path'}) to mono @ SAMPLE_RATE.

    Avoids the datasets Audio decoder (which now requires torchcodec); we read
    the embedded bytes with soundfile and resample with librosa ourselves.
    """
    if entry.get("bytes") is not None:
        wav, sr = sf.read(io.BytesIO(entry["bytes"]), dtype="float32", always_2d=False)
    else:
        wav, sr = sf.read(entry["path"], dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != C.SAMPLE_RATE:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=C.SAMPLE_RATE)
    return wav.astype(np.float32)


def build_label_map(ds) -> dict[int, str]:
    """Map integer label -> species name from the 'label' ClassLabel feature."""
    feat = ds.features["label"]
    if hasattr(feat, "names"):
        return {i: name for i, name in enumerate(feat.names)}
    mapping: dict[int, str] = {}
    for lab, sp in zip(ds["label"], ds["species"]):
        mapping[int(lab)] = str(sp)
    return dict(sorted(mapping.items()))


def process_split(ds) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for ex in tqdm(ds, desc="spectrograms"):
        wav = decode_audio(ex["audio"])
        X.append(logmel(wav))
        y.append(int(ex["label"]))
    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.int64)


def main() -> None:
    print(f"Loading {C.HF_DATASET} ...")
    dsd = load_dataset(C.HF_DATASET)
    print(dsd)

    # Keep audio as raw bytes; we decode + resample ourselves (no torchcodec).
    for split in dsd:
        dsd[split] = dsd[split].cast_column("audio", Audio(decode=False))

    train_split = "train"
    test_split = "test" if "test" in dsd else "validation"

    labels = build_label_map(dsd[train_split])
    C.LABELMAP.write_text(json.dumps(labels, indent=2))
    print(f"{len(labels)} classes -> {C.LABELMAP}")

    for split, out in [(train_split, C.CACHE_DIR / "train.npz"),
                       (test_split, C.CACHE_DIR / "test.npz")]:
        print(f"\nProcessing '{split}' ({len(dsd[split])} clips)")
        X, y = process_split(dsd[split])
        np.savez_compressed(out, X=X, y=y)
        print(f"  saved {X.shape} -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
