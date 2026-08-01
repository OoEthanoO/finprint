"""Silence and noise must not come back as a confident species.

The classifier is a closed set: it names its nearest match for any input, and it
is *more* confident on degenerate input than on a real quiet call (pure silence
scores ~0.72, white noise ~0.95). The 0.5 low-confidence line therefore cannot
catch this, so `finprint.quality` judges the audio itself. These tests pin both
directions: degenerate input is flagged, and real calls are not.
"""
import numpy as np
import pytest

import finprint.config as C
from finprint.features import extract
from finprint.quality import assess
from signals import pulse_train, tone, white_noise

SR = C.SAMPLE_RATE


def warn(wav):
    return assess(wav, extract(wav))


def test_pure_silence_is_flagged():
    w = np.zeros(SR * 2, dtype=np.float32)
    assert warn(w)["code"] == "silent"


def test_constant_dc_is_flagged_as_silent():
    """A DC offset has energy but no audio; naive RMS would let it through."""
    w = np.full(SR * 2, 0.4, dtype=np.float32)
    assert warn(w)["code"] == "silent"


def test_empty_input_is_flagged():
    assert assess(np.array([], dtype=np.float32), extract(tone(500, 1.0)))["code"] == "silent"


def test_white_noise_is_flagged_as_noise():
    assert warn(white_noise(2.0, amp=0.3))["code"] == "noise"


@pytest.mark.parametrize("freq", [200.0, 800.0, 3000.0])
def test_tonal_calls_are_not_flagged(freq):
    assert warn(tone(freq, 1.5)) is None


def test_pulse_train_is_not_flagged():
    """Burst pulses are broadband — the noise gate must not swallow them."""
    assert warn(pulse_train(4000.0, 60.0, 1.5)) is None


def test_quiet_but_real_call_is_not_flagged():
    """Well below the quietest real clip's level, but still audible structure."""
    assert warn(tone(900.0, 1.5) * 1e-3) is None


def test_predict_reports_the_warning(tmp_path):
    import soundfile as sf

    import finprint.predict as P

    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(SR * 2, dtype=np.float32), SR)
    out = P.predict(str(path), with_spectrogram=False)
    assert out["signal_warning"]["code"] == "silent"
    # the key must exist (and be None) for good audio, so clients can rely on it
    good = tmp_path / "tone.wav"
    sf.write(good, tone(900.0, 1.5), SR)
    assert P.predict(str(good), with_spectrogram=False)["signal_warning"] is None
