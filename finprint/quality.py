"""Does this upload actually contain an animal call?

The classifier is a closed-set softmax over 32 species: hand it silence and it
still returns its nearest match, and it does so *confidently* — pure silence
scores 0.72 and white noise 0.95, well above the 0.5 line the app uses to mark a
prediction unreliable. Confidence cannot catch this, because the failure is that
there is nothing to classify, not that the answer is uncertain.

So the input is checked directly, before its species is believed. Both thresholds
come from measurement against the held-out split (n=40 clips):

                      real clips        silence / DC      white noise
  RMS (DC removed)    min 5.6e-03       0.0 / 6.0e-08     3.0e-01
  spectral flatness   max 0.29          1.00 / 0.99       0.57

`SILENT_RMS` sits ~560x below the quietest real clip and far above a constant
signal, so it separates "no audio" from "quiet audio" with room to spare.
`NOISE_FLATNESS` is deliberately closer to the noise end than to the real-clip
end: wrongly flagging a genuine echolocation click train (broadband by nature,
and the closest real calls get to noise at 0.29) would look far more broken in
front of a user than missing a noise upload.
"""

from __future__ import annotations

import numpy as np

from .features import AcousticFeatures

SILENT_RMS = 1e-5        # below this there is effectively no audio
NOISE_FLATNESS = 0.45    # above this the spectrum is noise-like, not a call


def assess(wav: np.ndarray, feats: AcousticFeatures) -> dict | None:
    """Return a warning describing why this isn't a call, or None if it looks fine.

    Reuses the spectral flatness already measured during feature extraction, so
    this costs a mean and a dot product on top of what the request does anyway.
    """
    w = np.asarray(wav, dtype=np.float32).ravel()
    if w.size == 0:
        return {"code": "silent", "message": "This file contains no audio."}

    centred = w - float(w.mean())
    rms = float(np.sqrt(np.mean(centred * centred)))
    if rms < SILENT_RMS:
        return {
            "code": "silent",
            "message": "No audible audio — this clip is silent or a constant "
                       "tone-free signal, so there is no call to identify.",
        }

    if feats.spectral_flatness > NOISE_FLATNESS:
        return {
            "code": "noise",
            "message": "This sounds like broadband noise rather than a "
                       "structured call, so the species result is unreliable.",
        }

    return None
