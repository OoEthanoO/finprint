"""The degradations behind the robustness table must be what they say they are.

reports/error_analysis.md tells a reader not to play a call through a speaker at
a demo, and quantifies how bad 0 dB SNR is. Those claims are only worth anything
if "0 dB SNR" really is 0 dB and the band-limiter really removes the band it
names — a transform that quietly did less than advertised would make the model
look more robust than it is.
"""
import shutil

import numpy as np
import pytest

import finprint.config as C
from finprint.degrade import (CONDITIONS, add_noise, band_limit, mp3_roundtrip,
                              reverb, signal_rms, speaker_playback)
from signals import tone


def _rng():
    return np.random.default_rng(0)


def _band_energy(wav, lo, hi, sr=C.SAMPLE_RATE):
    spec = np.abs(np.fft.rfft(wav.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(wav), 1.0 / sr)
    sel = (freqs >= lo) & (freqs < hi)
    return float(np.sum(spec[sel] ** 2))


@pytest.mark.parametrize("snr_db", [20.0, 10.0, 0.0])
def test_added_noise_lands_within_a_fraction_of_a_dB_of_the_target(snr_db):
    """The headline of the worst row is its SNR, so it has to be measured."""
    clean = tone(1000.0, 4.0)
    noisy = add_noise(clean, snr_db, _rng())

    noise = noisy - clean
    achieved = 20.0 * np.log10(signal_rms(clean) / np.sqrt(np.mean(noise ** 2)))
    assert achieved == pytest.approx(snr_db, abs=0.5)


def test_snr_reference_is_the_call_not_the_surrounding_silence():
    """A mostly-silent clip must get the same SNR as a wall-to-wall one."""
    call = tone(1000.0, 1.0)
    padded = np.concatenate([np.zeros(C.SAMPLE_RATE * 8, dtype=np.float32), call])

    # Whole-clip RMS differs ~3x between these; the loudest-window RMS must not.
    assert signal_rms(padded) == pytest.approx(signal_rms(call), rel=0.05)
    assert np.sqrt(np.mean(padded ** 2)) < 0.5 * np.sqrt(np.mean(call ** 2))


def test_noise_is_reproducible_for_a_given_seed():
    clean = tone(800.0, 1.0)
    a = add_noise(clean, 10.0, np.random.default_rng(7))
    b = add_noise(clean, 10.0, np.random.default_rng(7))
    c = add_noise(clean, 10.0, np.random.default_rng(8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_band_limit_removes_energy_outside_the_band_and_keeps_it_inside():
    inside = tone(2000.0, 2.0)
    below = tone(100.0, 2.0)
    above = tone(12000.0, 2.0)

    assert _band_energy(band_limit(inside, 300.0, 8000.0), 300.0, 8000.0) > \
        0.5 * _band_energy(inside, 300.0, 8000.0)
    for out_of_band in (below, above):
        kept = np.sum(band_limit(out_of_band, 300.0, 8000.0).astype(np.float64) ** 2)
        assert kept < 0.01 * np.sum(out_of_band.astype(np.float64) ** 2)


def test_band_limit_leaves_a_too_short_clip_alone_rather_than_raising():
    tiny = tone(1000.0, 0.0005)
    assert np.array_equal(band_limit(tiny, 300.0, 8000.0), tiny)


def test_reverb_smears_a_click_forward_in_time():
    """The point of the speaker simulation: a transient stops being a transient."""
    click = np.zeros(C.SAMPLE_RATE, dtype=np.float32)
    click[100] = 1.0
    wet = reverb(click, rng=_rng())

    tail = np.sum(np.abs(wet[200:]))
    assert tail > 0.0
    assert np.sum(np.abs(click[200:])) == 0.0


needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not on PATH — the mp3 rows of the robustness table need it",
)


@needs_ffmpeg
def test_mp3_roundtrip_returns_comparable_audio_at_the_same_rate():
    clean = tone(1000.0, 1.0)
    out = mp3_roundtrip(clean, 64)
    # Encoder delay changes the length slightly; the content must survive.
    assert abs(len(out) - len(clean)) < C.SAMPLE_RATE * 0.2
    assert _band_energy(out, 500.0, 1500.0) > 10.0 * _band_energy(out, 5000.0, 15000.0)


@needs_ffmpeg
def test_lower_bitrate_discards_more():
    """32 kbps must be a harsher condition than 64, or the table's order is a lie."""
    clean = tone(1000.0, 1.0) + 0.3 * tone(9000.0, 1.0)
    hi = _band_energy(mp3_roundtrip(clean, 64), 8000.0, 16000.0)
    lo = _band_energy(mp3_roundtrip(clean, 32), 8000.0, 16000.0)
    assert lo < hi


def test_speaker_playback_is_strictly_harsher_than_its_parts():
    clean = tone(2000.0, 2.0)
    out = speaker_playback(clean, rng=_rng())
    # it band-limits ...
    assert _band_energy(out, 12000.0, 16000.0) < 0.05 * _band_energy(out, 300.0, 8000.0)
    # ... and it leaves noise behind
    assert signal_rms(out - band_limit(clean, 300.0, 8000.0)) > 0.0


@needs_ffmpeg   # two of the conditions are mp3 round-trips
def test_every_condition_returns_usable_audio_of_the_right_dtype():
    clean = tone(1500.0, 2.0)
    for name, fn in CONDITIONS.items():
        out = np.asarray(fn(clean, _rng()))
        assert out.ndim == 1 and out.size > 0, name
        assert np.isfinite(out).all(), name
        assert out.dtype == np.float32, name


def test_clean_is_the_identity_so_the_baseline_row_is_the_baseline():
    clean = tone(1500.0, 2.0)
    assert np.array_equal(CONDITIONS["clean"](clean, _rng()), clean)
