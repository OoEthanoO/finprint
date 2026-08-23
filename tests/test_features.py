"""Acoustic-feature extraction, including the FFT autocorrelation that replaced
the O(N^2) form. The equivalence test is a regression guard: if anyone reverts
_autocorrelate to np.correlate (or breaks the zero-padding), it fails here rather
than by silently hanging on a long clip in production.
"""
import numpy as np

import finprint.config as C
from finprint.features import _autocorrelate, _pulse_rate, extract, snr_db
from signals import pulse_train, tone, white_noise

SR = C.SAMPLE_RATE


def test_autocorrelate_matches_direct_form():
    rng = np.random.default_rng(3)
    for n in (1, 2, 17, 256, 4001):
        env = rng.standard_normal(n).astype(np.float32)
        ref = np.correlate(env.astype(np.float64), env.astype(np.float64),
                           mode="full")[n - 1:]
        got = _autocorrelate(env)
        assert got.shape == (n,)
        np.testing.assert_allclose(got, ref, rtol=1e-6,
                                   atol=1e-6 * max(1.0, np.abs(ref).max()))


def test_extract_duration_is_whole_clip_but_analysis_is_windowed():
    # 8 s, audible throughout: a quiet 500 Hz half then a louder 2 kHz half.
    quiet = tone(500, 4.0, amp=0.2, seed=1)
    loud = tone(2000, 4.0, amp=0.8, seed=2)
    f = extract(np.concatenate([quiet, loud]))
    assert f.duration_s > 6.0                                  # whole ~8 s clip
    assert abs(f.dominant_freq_hz - 2000) < abs(f.dominant_freq_hz - 500)


def test_extract_recovers_dominant_frequency_of_a_tone():
    f = extract(tone(1200, 1.0, amp=0.7, noise=0.005, seed=4))
    assert 1000 < f.dominant_freq_hz < 1500


def test_extract_returns_rounded_scalar_fields():
    f = extract(tone(800, 1.5, amp=0.6, seed=5))
    d = f.as_dict()
    assert set(d) >= {"duration_s", "dominant_freq_hz", "pulse_rate_hz",
                      "voiced_fraction"}
    assert all(isinstance(v, float) for v in d.values())


def test_pulse_rate_is_zero_for_broadband_noise():
    assert _pulse_rate(white_noise(3.0, amp=0.4, seed=6), SR) == 0.0


def test_pulse_rate_detects_a_known_modulation_rate():
    r = _pulse_rate(pulse_train(2000, 20.0, 3.0), SR)
    assert 18.0 < r < 22.0                                     # ~20 pulses/s


# --- blind SNR estimate ----------------------------------------------------
# The estimate has one job: order clips by how far the call rises above the
# background. These pin that ordering and the rough calibration, not exact
# values -- the number is a measurement of the clip, and a threshold that reads
# it belongs with real clips, not here.

def test_snr_estimate_tracks_added_noise():
    """A call at a known SNR reads back at roughly that SNR."""
    from finprint.degrade import add_noise

    call = tone(6000, 1.5)
    for target in (20.0, 10.0, 0.0):
        rng = np.random.default_rng(7)
        got = snr_db(add_noise(call, target, rng))
        assert abs(got - target) < 4.0, f"{target} dB read as {got}"


def test_snr_estimate_is_high_for_a_clean_call_and_low_for_noise():
    assert snr_db(tone(3000, 1.5)) > 30.0
    assert snr_db(white_noise(2.0)) < 0.0


def test_snr_estimate_survives_a_call_with_no_quiet_frames():
    """The reason the estimate is spectral rather than temporal.

    `extract` trims silence and analyses the loudest window, so a clean clip
    arrives as very nearly all call. An estimator that read its noise floor from
    the quiet *frames* would find only call there and score a pristine recording
    as buried -- which is exactly backwards.
    """
    continuous = tone(5000, 4.0)          # fills the whole analysis window
    assert snr_db(continuous) > 30.0


def test_snr_estimate_is_bounded_on_degenerate_input():
    """No inf, no NaN -- this is serialised into the API response."""
    for w in (np.zeros(SR, dtype=np.float32),
              np.full(SR, 0.4, dtype=np.float32),
              np.array([], dtype=np.float32)):
        v = snr_db(w)
        assert np.isfinite(v) and -31.0 <= v <= 61.0


def test_snr_db_matches_the_value_extract_reports():
    """The cheap path and the full path must not drift apart."""
    w = tone(2200, 1.5, noise=0.01, seed=5)
    assert abs(snr_db(w) - extract(w).snr_db) < 0.05
