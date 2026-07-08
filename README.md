# finprint 🐋

**Marine-mammal call classifier.** Upload or record an audio clip; finprint
returns the most likely **species** (a trained CNN) plus the **call type** and
the **acoustic measurements** behind that call (a transparent signal-analysis
layer), all in a small web app.

---

## What it predicts (and an honest note)

| Output | How | Trained? |
|---|---|---|
| **Species** (32 marine mammals) | CNN over log-mel spectrograms | ✅ yes, on the WMMS dataset |
| **Call type** (click / burst-pulse / whistle / song-moan / broadband) | rules over measured acoustic features | ➖ signal analysis, not a trained model |
| **Acoustic features** (dominant freq, bandwidth, f0 contour, pulse rate, …) | librosa DSP | ➖ direct measurement |

We originally wanted to predict **behavioral context** (mating, hunting, …).
After surveying the data, that isn't honestly possible: the Watkins database —
and essentially every at-scale, multi-species marine-mammal dataset — has **no
behavioral or call-type labels**. So instead of fabricating a "behavior" model,
finprint classifies calls by their **acoustic structure**, which is real,
measurable, and how bioacousticians actually categorize calls. Every call-type
prediction ships with the numbers and the reason it was chosen.

---

## Architecture

```
audio file ──▶ load @ 32 kHz, mono
           │
           ├─▶ highest-energy 4 s window ─▶ log-mel spectrogram ─▶ CNN ─▶ species (top-3)
           │
           └─▶ librosa DSP ─▶ acoustic features ─▶ rule-based classifier ─▶ call type
```

- **Species model** — 4-block conv net over 128-bin log-mel spectrograms, global
  average pooling, class-weighted cross-entropy, SpecAugment, early stopping on
  validation macro-F1. Small on purpose (~1.4k training clips).
- **Call-type layer** — [`finprint/features.py`](finprint/features.py) measures
  the clip; [`finprint/calltype.py`](finprint/calltype.py) scores it against
  each acoustic category with transparent, documented thresholds.

---

## Dataset

[**Watkins Marine Mammal Sound Database**](https://huggingface.co/datasets/confit/wmms-parquet)
(`confit/wmms-parquet`), Woods Hole Oceanographic Institution & New Bedford
Whaling Museum — **32 species**, ~1,357 train / 340 test clips.

The data is heterogeneous: native sample rates span **600 Hz – 166 kHz** and
durations **0.2 s – 114 s**. finprint normalizes this by resampling to 32 kHz and
extracting the loudest 4-second window per clip. *Academic use only.*

---

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# 1. download + cache spectrograms   (~data/cache/*.npz, models/labels.json)
python -m scripts.prepare_data

# 2. train the species CNN           (~models/species_cnn.pt)
python -m finprint.train

# 3. evaluate on the test split      (~reports/*)
python -m finprint.evaluate

# 4. launch the web app  ->  http://127.0.0.1:8000
uvicorn app.main:app --reload
```

The web app works without step 2 — you'll still get call type, features, and the
spectrogram; species prediction simply stays disabled until a model is trained.

---

## Results

Held-out **WMMS test split** (340 clips, 32 species), from
`python -m finprint.evaluate` — full per-class report and confusion matrix in
[reports/](reports/).

| Metric | Test |
|---|---|
| Accuracy | **0.794** |
| Top-3 accuracy | **0.915** |
| Macro-F1 | **0.802** |

Trained in ~2 min on an Apple-Silicon GPU (MPS), 60 epochs, best model chosen by
validation macro-F1 (0.894).

---

## Project layout

```
finprint/
├── finprint/          core package (audio, model, features, calltype, train, evaluate, predict)
├── app/               FastAPI backend + single-page UI
├── scripts/           prepare_data.py
├── data/              downloaded dataset + cached spectrograms  (gitignored)
├── models/            trained checkpoint + label map            (gitignored)
└── reports/           metrics + confusion matrix
```

## Limitations & next steps

- **No behavior labels** exist, so call type is acoustic structure, not context.
- Resampling to 32 kHz discards ultrasonic click energy above 16 kHz.
- Small dataset — a pretrained audio encoder (AST/PANNs) would likely lift
  accuracy; SpecAugment + class weighting already help.
- Call-type thresholds are literature-informed heuristics, not fit to labels.
