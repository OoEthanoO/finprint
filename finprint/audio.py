"""Shared audio -> log-mel-spectrogram transform.

Both the offline data-prep step and the live web app call into here, so a clip
uploaded to the app is turned into *exactly* the same features the model saw at
training time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import config as C

if TYPE_CHECKING:                     # annotation only; never imported at runtime
    import torchaudio

# One global transform, built lazily so importing this module stays cheap.
# torch/torchaudio are imported inside the two functions that need them, for the
# same reason: the waveform helpers below are plain numpy and are shared with the
# DSP layer, which should not have to load torch to measure a clip.
_MEL = None


def _mel_transform() -> torchaudio.transforms.MelSpectrogram:
    global _MEL
    if _MEL is None:
        import torchaudio

        _MEL = torchaudio.transforms.MelSpectrogram(
            sample_rate=C.SAMPLE_RATE,
            n_fft=C.N_FFT,
            hop_length=C.HOP_LENGTH,
            n_mels=C.N_MELS,
            f_min=C.F_MIN,
            f_max=C.F_MAX,
            power=2.0,
        )
    return _MEL


def loudest_window(wav: np.ndarray, n_samples: int) -> np.ndarray:
    """Highest-RMS-energy window of n_samples, or the whole clip if it is shorter.

    Recordings run up to ~2 minutes and are often mostly silence, so a blind
    center-crop would frequently miss the call. We slide a window at a coarse
    stride and keep the one with the highest RMS energy.

    Unlike fix_length() this never pads: callers that *measure* the waveform
    must not have invented silence folded into their statistics.
    """
    wav = np.asarray(wav, dtype=np.float32).ravel()
    if len(wav) <= n_samples:
        return wav

    best_start, best_energy = 0, -1.0
    for start in range(0, len(wav) - n_samples + 1, C.ENERGY_WIN_STRIDE):
        seg = wav[start:start + n_samples]
        energy = float(np.mean(seg * seg))
        if energy > best_energy:
            best_energy, best_start = energy, start
    return wav[best_start:best_start + n_samples]


def fix_length(wav: np.ndarray, n_samples: int = C.N_SAMPLES) -> np.ndarray:
    """Return a fixed-length window: pad short clips, else take the loudest window."""
    wav = np.asarray(wav, dtype=np.float32).ravel()
    if len(wav) <= n_samples:
        pad = n_samples - len(wav)
        left = pad // 2
        return np.pad(wav, (left, pad - left))
    return loudest_window(wav, n_samples)


def peak_normalize(wav: np.ndarray) -> np.ndarray:
    """Scale so the loudest sample is +-1. Removes recording-gain differences."""
    wav = np.asarray(wav, dtype=np.float32)
    peak = np.max(np.abs(wav)) if wav.size else 0.0
    if peak > 1e-9:
        wav = wav / peak
    return wav


def logmel(wav: np.ndarray) -> np.ndarray:
    """Waveform (mono, SAMPLE_RATE) -> log-mel spectrogram, shape [N_MELS, N_FRAMES]."""
    import torch

    wav = fix_length(peak_normalize(wav))
    t = torch.from_numpy(wav).float().unsqueeze(0)          # [1, N_SAMPLES]
    mel = _mel_transform()(t)                               # [1, N_MELS, T]
    logm = torch.log(mel + 1e-6).squeeze(0)                 # [N_MELS, T]
    return logm.numpy().astype(np.float32)
