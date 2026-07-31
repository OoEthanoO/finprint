"""Windowing invariants for the shared audio layer.

The split between loudest_window (never pads) and fix_length (pads short clips)
is what lets the DSP layer measure real signal while the CNN still gets a fixed
tensor — so both halves of that contract are pinned here.
"""
import numpy as np

import finprint.config as C
from finprint.audio import fix_length, logmel, loudest_window
from signals import tone, white_noise


def test_loudest_window_never_pads_short_clip():
    w = tone(1000, 0.5)                       # shorter than the analysis window
    out = loudest_window(w, C.ANALYSIS_SAMPLES)
    assert out.shape[0] == len(w)             # returned whole, no invented silence
    np.testing.assert_array_equal(out, w)


def test_loudest_window_returns_exact_length_for_long_clip():
    out = loudest_window(tone(1000, 10.0), C.ANALYSIS_SAMPLES)
    assert out.shape[0] == C.ANALYSIS_SAMPLES


def test_loudest_window_lands_on_the_loud_region():
    quiet = white_noise(10.0, amp=0.01, seed=1)
    burst = tone(1500, C.ANALYSIS_SECONDS, amp=0.9, seed=2)
    start = 5 * C.SAMPLE_RATE                  # a multiple of ENERGY_WIN_STRIDE
    w = quiet.copy()
    w[start:start + len(burst)] = burst
    out = loudest_window(w, C.ANALYSIS_SAMPLES)
    rms = float(np.sqrt(np.mean(out ** 2)))
    assert rms > 0.5                           # burst RMS ~0.64; quiet ~0.01


def test_fix_length_tiles_short_clip_without_inventing_audio():
    """Short clips are filled by repeating them, not padded with silence.

    Zero padding left the median clip as mostly silence and cost ~10 points of
    accuracy; tiling repeats the clip's own samples, so the window starts with
    the original signal and contains no invented content.
    """
    w = tone(1000, 0.5)
    out = fix_length(w)
    assert out.shape[0] == C.N_SAMPLES
    # begins with the untouched clip
    np.testing.assert_array_equal(out[:len(w)], w)
    # every sample comes from the clip itself, and none of it is silence
    assert np.isin(out, w).all()
    assert np.abs(out).sum() > 0.0
    # the repeat is periodic at the clip length
    np.testing.assert_array_equal(out[len(w):2 * len(w)], w)


def test_fix_length_handles_empty_input():
    assert fix_length(np.array([], dtype=np.float32)).shape[0] == C.N_SAMPLES


def test_fix_length_windows_long_clip():
    assert fix_length(tone(1000, 10.0)).shape[0] == C.N_SAMPLES


def test_logmel_has_model_input_shape():
    mel = logmel(tone(1000, 5.0))
    assert mel.shape == (C.N_MELS, C.N_FRAMES)
