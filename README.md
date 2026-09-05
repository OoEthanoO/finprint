---
title: finprint
emoji: 🐋
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

<!-- The YAML above is Hugging Face Space metadata, and it has to stay the very
first thing in this file. GitHub and HF only treat a fenced block as frontmatter
when nothing at all precedes it — put anything above it, even a comment like this
one, and GitHub instead renders a horizontal rule and a heading made of the raw
YAML, while HF stops seeing the Space config. -->

# finprint 🐋

**Marine-mammal call classifier.** Upload or record an audio clip; finprint
returns the most likely **species** (a trained CNN) plus the **call type** and
the **acoustic measurements** behind that call (a transparent signal-analysis
layer), all in a small web app.

---

## What it predicts (and an honest note)

| Output | How | Trained? |
|---|---|---|
| **Group** (toothed whale / baleen whale / pinniped) | the top species' family, with its summed probability | ✅ derived from the CNN — **0.979** on held-out data |
| **Species** (32 marine mammals) | CNN over log-mel spectrograms | ✅ yes, on the WMMS dataset — 0.906 |
| **Call type** (click / burst-pulse / whistle / song-moan / broadband) | rules over measured acoustic features | ➖ signal analysis, not a trained model |
| **Acoustic features** (dominant freq, bandwidth, f0 contour, pulse rate, signal-to-noise, …) | librosa DSP | ➖ direct measurement |

The app leads with the **group** because the model's mistakes stay inside the
family — it confuses one dolphin for another, not a dolphin for a seal. So the
group is right ~98% of the time even when the species underneath it is wrong.

We originally wanted to predict **behavioral context** (mating, hunting, …).
After surveying the data, that isn't honestly possible: the Watkins database —
and essentially every at-scale, multi-species marine-mammal dataset — has **no
behavioral or call-type labels**. So instead of fabricating a "behavior" model,
finprint classifies calls by their **acoustic structure**, which is real,
measurable, and how bioacousticians actually categorize calls. Every call-type
prediction ships with the numbers and the reason it was chosen.

**Species is always one of the 32.** The CNN can only name a species it was
trained on, so a clip from any other animal (a blue whale, say) still comes back
as its nearest match. When the top score is below **0.5** — the point where
held-out accuracy falls from **0.97 to 0.49** — the app labels the prediction
*low confidence* instead of presenting it as certain. (The group above it stays
reliable even there, which is the other reason to show it.)

**Silence and noise are rejected outright.** Confidence alone cannot catch them:
a closed-set softmax is *more* certain on garbage than on a quiet real call —
pure silence scores 0.72 and white noise 0.95, both above that 0.5 line. So
[`finprint/quality.py`](finprint/quality.py) checks the audio itself and the app
says there is nothing to identify instead of naming a whale. The thresholds are
set from measurement, deliberately biased against false alarms on genuine
broadband calls like echolocation click trains.

---

## Architecture

```
audio file ──▶ load @ 32 kHz, mono ──▶ highest-energy 4 s window
                                   │
                                   ├─▶ log-mel spectrogram ─▶ CNN ─▶ species (top-3)
                                   │
                                   └─▶ librosa DSP ─▶ acoustic features ─▶ call type (rules)
```

Both branches read the same 4 s window, so the species, the call type, and the
printed numbers all describe one call rather than two minutes of ocean. Reported
*duration* is the exception: it measures the whole trimmed recording.

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

Optionally, measure what survives degraded audio — mp3, noise at a given SNR,
and a simulated speaker-played-then-phone-recorded clip (`~reports/robustness.json`):

```bash
python -m scripts.robustness          # needs ffmpeg on PATH
```

The web app works without step 2 — you'll still get call type, features, and the
spectrogram; species prediction simply stays disabled until a model is trained.

The API caps uploads at **25 MB** and decodes at most the **first 10 minutes** of
audio, so one large file can't exhaust the small serving instance
(see [`finprint/config.py`](finprint/config.py)).

## Tests

```bash
pytest tests/
```

Covers the windowing invariants, the FFT-autocorrelation equivalence (a guard
against the O(N²) form that used to hang on long clips), the call-type rules, and
the HTTP contract including the upload guard.

---

## Deploy

**Live: <https://finprint.ethanyanxu.com>** — self-hosted on a Windows laptop
behind Caddy, serving both the API and this page. [**DEPLOY.md**](DEPLOY.md) is
the canonical guide (setup and update scripts, the Fly.io and Render configs kept
in-repo, and how to run the exact image locally).

The whole app is one process: FastAPI serves `/api/*` and the static page
together, so the frontend is same-origin wherever it runs — behind a reverse
proxy on a custom domain, a local `uvicorn`, or a container host. Static/edge platforms (Vercel,
Cloudflare Pages) can't host the *backend* — it needs native libs, ffmpeg, and
more than an edge runtime allows — but they can host the page on its own if you
ever want to split them:

- Set `API_BASE` in [app/static/index.html](app/static/index.html) to the backend
  URL and deploy `app/static` as a static site. Leave it empty for the normal
  single-container setup.
- CORS is already enabled on the API — `*` by default, restrict it with the
  `FINPRINT_ALLOW_ORIGINS` env var.

**Hugging Face Spaces** works too, as an alternative host for the same image (the
Space metadata at the top of this file configures it):

```bash
# needs a free HF account + write token: https://huggingface.co/settings/tokens
HF_TOKEN=hf_xxx SPACE_ID=<username>/finprint python -m scripts.deploy_hf
```

---

## Results

Held-out **WMMS test split** (340 clips, 32 species), from
`python -m finprint.evaluate` — full per-class report and confusion matrix in
[reports/](reports/).

| Metric | Test |
|---|---|
| Accuracy | **0.906** |
| Top-3 accuracy | **0.965** |
| Macro-F1 | **0.878** |
| Group (toothed whale / baleen whale / pinniped) | **0.979** |
| Accuracy among predictions shown as answers (top score ≥ 0.5) | **0.973** |

The last two rows are what the UI actually puts in front of a user, so
`finprint.evaluate` computes them alongside species accuracy and writes all of it
to [reports/metrics.json](reports/metrics.json) — including the group-confidence
calibration table. Nothing quoted here is measured by hand, so a retrain cannot
leave these numbers describing an older model.

Trained in ~2 min on an Apple-Silicon GPU (MPS), 60 epochs, best model chosen by
validation macro-F1 (0.957).

Accuracy rose from 0.788 to 0.906 by filling short clips with a repeat of the
clip instead of silence — the median clip is ~1.7 s against a 4 s input, so a
typical example used to be ~60% padding. [reports/error_analysis.md](reports/error_analysis.md)
has the variant comparison, the error breakdown, and one approach that was
measured and rejected.

---

## Project layout

```
finprint/
├── finprint/          core package (audio, model, features, calltype, train, evaluate, predict)
├── app/               FastAPI backend + single-page UI
├── tests/             pytest suite (DSP, call type, API)
├── scripts/           prepare_data.py, robustness.py
├── data/              downloaded dataset + cached spectrograms  (gitignored)
├── models/            trained checkpoint (committed, ~2.3 MB) + label/norm JSON
└── reports/           metrics + confusion matrix
```

## Limitations & next steps

- **No behavior labels** exist, so call type is acoustic structure, not context.
- **Trained only on clean archive audio.** Compression is free (64 kbps mp3 costs
  nothing measurable), but re-recording through the air is not — playing a call
  through a speaker and capturing it on a phone drops species accuracy to 0.14.
  Upload files directly. The group label survives it far better (0.74), and the
  app declines to answer at all on 99% of those clips, which is the right
  behaviour. Regenerate the full table with `python -m scripts.robustness`; it is
  reproduced in [reports/error_analysis.md](reports/error_analysis.md).
- **Unreliable at 0 dB SNR** — noise as loud as the call. This is the worst case,
  not the speaker one: the app still presents ~22% of those clips as answers and
  is right on 30% of them, so it is confidently wrong often enough to matter. The
  silence/noise gate does not catch it, because a buried call is still a call.
- Resampling to 32 kHz discards ultrasonic click energy above 16 kHz.
- Small dataset — a pretrained audio encoder (AST/PANNs) would likely lift
  accuracy; SpecAugment + class weighting already help.
- Call-type thresholds are literature-informed heuristics, not fit to labels.
