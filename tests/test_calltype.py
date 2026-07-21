"""Structural call-type rules. These lock in the intended mapping from acoustic
structure to category so a threshold tweak can't silently reclassify calls."""
from finprint.calltype import CATEGORIES, classify
from finprint.features import extract
from signals import tone, white_noise


def test_low_frequency_sustained_tone_is_song_moan():
    f = extract(tone(180, 5.0, amp=0.7, seed=1))
    assert classify(f).label == "Song / moan"


def test_high_frequency_tonal_call_is_whistle():
    f = extract(tone(3000, 2.0, amp=0.7, seed=2))
    assert classify(f).label == "Whistle"


def test_broadband_noise_reads_as_click_or_grunt():
    f = extract(white_noise(1.5, amp=0.4, seed=3))
    assert classify(f).label in {"Echolocation click", "Broadband pulse / grunt"}


def test_scores_form_a_normalized_distribution():
    c = classify(extract(tone(180, 5.0, amp=0.7, seed=4)))
    assert set(c.scores) == set(CATEGORIES)
    assert abs(sum(c.scores.values()) - 1.0) < 0.05     # 3-dp rounding slack
    assert 0.0 <= c.confidence <= 1.0
    assert c.label in CATEGORIES
    assert c.rationale                                   # non-empty explanation
