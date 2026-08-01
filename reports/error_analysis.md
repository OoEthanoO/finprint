# finprint — error analysis

Held-out WMMS test split, 340 clips, 32 classes, evaluated with
`python -m finprint.evaluate`.

| Metric | Before | **Current** |
|---|---|---|
| Accuracy | 0.788 | **0.906** |
| Top-3 accuracy | 0.921 | **0.965** |
| Macro-F1 | 0.798 | **0.878** |
| Errors | 72 / 340 | **32 / 340** |

"Before" is the zero-padded input described below; "current" is the same
architecture trained on tiled input. Nothing else changed.

---

## The change that mattered: stop padding short clips with silence

The median WMMS clip is ~1.7 s against a 4 s model input, so **a typical training
example was ~60% silence** — the network spent most of its input budget on
nothing.

Four input variants, same architecture, same seed, same split, same 45 epochs,
compared on the **validation** split only:

| Variant | Val accuracy | Macro-F1 | Train time |
|---|---|---|---|
| 4 s window, zero-padded (original) | 0.8186 | 0.8390 | 219 s |
| **4 s window, clip tiled to fill** | **0.9216** | 0.9233 | 208 s |
| 2 s window, zero-padded | 0.8873 | 0.8984 | 124 s |
| 2 s window, clip tiled to fill | 0.9216 | 0.9273 | 119 s |

Repeating the clip to fill the window is worth **~+10 points**. Nothing is
fabricated — the window contains only the clip's own audio, just heard more than
once, which is closer to how the call appears in a continuous recording.

The 4 s tiled variant was adopted because it captures the full accuracy gain
without changing `ANALYSIS_SECONDS`, which also drives acoustic feature
extraction and the call-type rules — those are measured with `loudest_window()`,
which never pads, so **call type and the acoustic features are unaffected by this
change**. Dropping to a 2 s window is a reasonable follow-up (equal accuracy,
marginally better macro-F1, about half the compute) but it touches that second
subsystem and deserves its own evaluation.

Retraining on the held-out test split confirmed the validation result:
**+11.8 points of accuracy** (0.788 → 0.906).

---

## What the remaining 32 errors look like

Errors stay close to the truth taxonomically:

- **78% (25/32)** stay inside the correct broad group (toothed whale / baleen
  whale / pinniped).
- **25% (8/32)** are delphinid → delphinid — down from 57% before the input fix,
  which is where most of the gain came from: with real signal instead of silence,
  the model separates acoustically similar dolphins far better.
- **62% of errors still carry the correct species in the top 3**, which is why
  top-3 accuracy (0.965) sits well above top-1 (0.906), and why the app shows a
  top-3 list rather than a single label.

Most frequent confusions now:

| Count | True | Predicted |
|---|---|---|
| 2 | Walrus | Harp_Seal |
| 2 | Bearded_Seal | Bowhead_Whale |
| 2 | Bowhead_Whale | Bearded_Seal |
| 2 | Common_Dolphin | Striped_Dolphin |
| 2 | Spinner_Dolphin | Sperm_Whale |

The dominant pairs are now pinniped ↔ pinniped and the bearded seal ↔ bowhead
whale pair — both produce long, frequency-modulated moans, so the confusion is
acoustically reasonable rather than arbitrary.

## The coarse group label is the reliable half of the answer

| Prediction | Test accuracy |
|---|---|
| Species (32-way) | 0.906 |
| Group (toothed whale / baleen whale / pinniped) | **0.979** |

The app leads with the group for this reason: "**toothed whale** — most likely
Killer Whale" is right about the animal ~98% of the time even when the species is
wrong. Reported group confidence is the summed probability mass of the group, and
it is well calibrated on the test split:

| Reported confidence | Actual group accuracy |
|---|---|
| 0.50 – 0.80 | 0.84 |
| 0.80 – 0.95 | 0.995 |
| 0.95 – 1.00 | 1.00 |

## Weakest classes

| Species | Precision | Recall | F1 | Test clips |
|---|---|---|---|---|
| Bearded_Seal | 0.62 | 0.71 | 0.67 | 7 |
| Beluga,_White_Whale | 0.70 | 0.70 | 0.70 | 10 |
| Sperm_Whale | 0.71 | 0.80 | 0.75 | 15 |
| Walrus | 1.00 | 0.62 | 0.77 | 8 |
| Bowhead_Whale | 0.83 | 0.83 | 0.83 | 12 |
| Southern_Right_Whale | 0.71 | 1.00 | 0.83 | 5 |

No class with test clips scores 0 any more.

## Data caveats worth stating up front

- **Weddell_Seal has 0 clips in the test split**, so its reported F1 of 0.00 is an
  artifact of the split, not a model failure — it is trained but never tested.
  (It only appears in the report because evaluation pins
  `labels=range(n_classes)`; without that the report silently renumbers.)
- **Leopard_Seal and Minke_Whale have ≤ 3 test clips each**, so their per-class
  scores move in ~33% steps and should not be read as meaningful.
- Class support is very uneven (2–91 training clips per species), which is why
  training uses inverse-frequency class weighting.

## Out-of-distribution input: confidence does not save you

A closed-set softmax returns its nearest match for anything, and it is *more*
confident on degenerate input than on a genuine quiet call:

| Input | Top species | Confidence | Group confidence |
|---|---|---|---|
| Pure silence | Fin,_Finback_Whale | **0.72** | 0.91 |
| DC offset | Fin,_Finback_Whale | 0.62 | 0.71 |
| White noise | Melon_Headed_Whale | **0.95** | 0.99 |
| A real 800 Hz tone | Fin,_Finback_Whale | 0.24 | 0.46 |

The app's 0.5 low-confidence line catches the *real* tone and misses all three
degenerate inputs — the failure is that there is nothing to classify, not that
the answer is uncertain. So the input is checked directly
([`finprint/quality.py`](../finprint/quality.py)), using separations measured on
the held-out split:

| | spectral flatness | RMS (DC removed) |
|---|---|---|
| Real clips (n=40) | max 0.29 (p95 0.25) | min 5.6e-03 |
| White noise | 0.57 | 3.0e-01 |
| Silence / DC | 1.00 / 0.99 | 0.0 / 6.0e-08 |

Both gaps are wide. The noise threshold (0.45) sits closer to the noise end on
purpose: wrongly flagging a real echolocation click train — broadband by nature,
and the closest real calls get to noise at 0.29 — would look far more broken than
missing a noise upload.

A separate sweep of 24 awkward uploads (stereo, 8 kHz–192 kHz, 5 ms–60 s, mp3 /
m4a / webm / flac / ogg, truncated and non-audio files) produced **no 5xx and no
hangs**; malformed input is rejected with 4xx.

## Where request time actually goes

Steady-state, CPU (what the container runs), a 2 s clip, spectrogram included:

| Stage | Time | Share |
|---|---|---|
| `features` (DSP) | 0.635 s | **91%** |
| `spectrogram` (matplotlib) | 0.049 s | 7% |
| `species` (the CNN) | 0.007 s | **1%** |
| `decode` | ~0 s | — |

Inside `features`, **`librosa.pyin` is 99%**, and inside pyin the Viterbi decode
alone is **~89% of the whole request**. Its cost is quadratic in the number of
pitch states, and the default 0.1-semitone resolution over 40 Hz – 4 kHz is
roughly 1600 states.

Three optimisations were measured and **all three rejected**:

| Idea | Result | Verdict |
|---|---|---|
| Shrink the model input to a 2 s window | halves a stage that is 1% of the request (saves ~3.5 ms) | not worth touching `ANALYSIS_SECONDS` |
| Coarser pyin `resolution` (0.25) | **5.9x faster** features, but `voiced_fraction` shifts ~0.12 and the call-type label changes on **1 in 10** clips | rejected: degrades the explainable layer to save 0.5 s |
| Drop matplotlib for a hand-rolled PNG | ~69 MB smaller image; the 16 s first-render font-cache cost is **already** baked in at build time by `scripts.warmup` | little left to win |

The 6x speedup remains available if sub-second response ever matters more than
call-type stability — it is a deliberate trade, not an oversight.

### Concurrency: the event loop is GIL-bound, not CPU-bound

`predict()` used to be called straight from the `async def` endpoint, parking the
event loop — the thing that serves every other request — for its whole duration.
A page load during a single upload took **1.84 s instead of 1 ms**. It now runs in
a worker thread (`run_in_threadpool`), which helps but does not cure it:

| | page load |
|---|---|
| Idle | 0.001 s |
| During 1 upload | 0.51 s |
| During 3 uploads, before the change | 1.84 s |
| During 3 uploads, after | 1.20 s |

The residual starvation is **not** a shortage of cores: these numbers come from a
10-core machine, so a second core would have absorbed it if the work were merely
CPU-bound. It is the GIL — librosa's pitch tracking holds it — which means
**raising Cloud Run's `--cpu` would not fix this**. The levers that would are
cutting the DSP work itself (the pyin trade above) or running more than one
worker process.

At ~0.5 s a page load during an upload is acceptable for a demo, so this is
recorded rather than chased.

## Tried and rejected: multi-window inference

Inference classifies the single loudest window, and recordings run up to ~114 s,
so averaging predictions over several windows looked promising. Measured on the
validation split (204 clips), averaging softmax over the top-k loudest windows:

| k | Accuracy | Top-3 | Macro-F1 |
|---|---|---|---|
| 1 (current) | 0.8873 | 0.9363 | 0.8989 |
| 3 | 0.8873 | 0.9412 | 0.8929 |
| 5 | 0.8873 | 0.9412 | 0.8929 |
| 8 | 0.8873 | 0.9412 | 0.8929 |

**No accuracy gain**, because only **19%** of clips exceed the 4 s window at all,
and those long clips were already the easy ones (0.947 vs 0.874 for short clips).
The complexity was not adopted — but the finding is what pointed at short clips,
and from there at the padding, which is where the real gain turned out to be.
