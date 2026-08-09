"""The two metrics the app's UI is built on, measured on synthetic scores.

`finprint.evaluate` reports species accuracy, but the app leads with the broad
group and hides a species behind a confidence line — so those are the numbers
quoted in the README, and they are the ones that would silently go stale after a
retrain if nothing computed them. These tests pin the arithmetic against cases
whose answer is known by construction, so the reported figures can be trusted
without a trained checkpoint or the cached dataset.
"""
import numpy as np
import pytest

import finprint.config as C
from finprint.evaluate import confidence_metrics, group_metrics
from finprint.taxonomy import (BALEEN_WHALE, PINNIPED, TOOTHED_WHALE,
                               group_confidence)

# Two species per group, in a fixed column order, so a hand-written probability
# row has an obvious group reading.
NAMES = [
    "Killer_Whale", "Common_Dolphin",      # 0, 1 -> toothed whale
    "Humpback_Whale", "Minke_Whale",       # 2, 3 -> baleen whale
    "Walrus", "Harp_Seal",                 # 4, 5 -> pinniped
]


def _probs(rows) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64)


def test_group_confidence_sums_only_its_own_group():
    row = [0.4, 0.3, 0.1, 0.05, 0.1, 0.05]
    assert group_confidence(row, NAMES, TOOTHED_WHALE) == pytest.approx(0.7)
    assert group_confidence(row, NAMES, BALEEN_WHALE) == pytest.approx(0.15)
    assert group_confidence(row, NAMES, PINNIPED) == pytest.approx(0.15)


def test_group_confidence_over_all_groups_is_the_whole_distribution():
    row = [0.4, 0.3, 0.1, 0.05, 0.1, 0.05]
    total = sum(group_confidence(row, NAMES, g)
                for g in (TOOTHED_WHALE, BALEEN_WHALE, PINNIPED))
    assert total == pytest.approx(1.0)


def test_group_forgives_a_within_group_species_error():
    """The whole point of the group label: dolphin-for-dolphin is still right."""
    probs = _probs([
        [0.6, 0.3, 0.0, 0.0, 0.1, 0.0],   # predicts Killer_Whale
        [0.3, 0.6, 0.0, 0.0, 0.1, 0.0],   # predicts Common_Dolphin
    ])
    y = np.array([1, 1])                   # both are really Common_Dolphin

    assert (probs.argmax(1) == y).mean() == 0.5      # species: one wrong
    assert group_metrics(probs, y, NAMES)["accuracy"] == 1.0  # group: both right


def test_group_counts_a_cross_group_error_as_wrong():
    probs = _probs([[0.1, 0.1, 0.1, 0.1, 0.6, 0.0]])  # predicts Walrus
    y = np.array([0])                                  # really Killer_Whale
    assert group_metrics(probs, y, NAMES)["accuracy"] == 0.0


def test_group_calibration_bins_report_their_own_accuracy():
    """Each bin's accuracy must describe only the clips that landed in it."""
    probs = _probs([
        # group mass 0.98 -> the 0.95-1.0 bin, and correct
        [0.6, 0.38, 0.01, 0.0, 0.01, 0.0],
        # group mass 0.6 -> the 0.5-0.8 bin, and wrong (truth is a pinniped)
        [0.35, 0.25, 0.2, 0.0, 0.2, 0.0],
    ])
    y = np.array([0, 4])
    bins = {tuple(b["confidence_range"]): b
            for b in group_metrics(probs, y, NAMES)["calibration"]}

    assert bins[(0.95, 1.0)]["n"] == 1
    assert bins[(0.95, 1.0)]["accuracy"] == 1.0
    assert bins[(0.5, 0.8)]["n"] == 1
    assert bins[(0.5, 0.8)]["accuracy"] == 0.0


def test_group_calibration_covers_every_clip_exactly_once():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(len(NAMES)), size=50)
    y = rng.integers(0, len(NAMES), size=50)
    out = group_metrics(probs, y, NAMES)
    assert sum(b["n"] for b in out["calibration"]) == out["n"] == 50


def test_group_skips_species_it_has_no_group_for():
    """An unmapped species is excluded, not silently counted as an error."""
    names = list(NAMES)
    names[5] = "Unmapped_Species"
    probs = _probs([
        [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],   # mapped, correct
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],   # truth is the unmapped species
    ])
    y = np.array([0, 5])
    out = group_metrics(probs, y, names)
    assert out["n"] == 1
    assert out["accuracy"] == 1.0


def test_confidence_split_separates_presented_answers_from_flagged_ones():
    probs = _probs([
        [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],   # confident, correct
        [0.8, 0.2, 0.0, 0.0, 0.0, 0.0],   # confident, wrong
        [0.3, 0.25, 0.2, 0.15, 0.1, 0.0],  # flagged, correct
        [0.3, 0.25, 0.2, 0.15, 0.1, 0.0],  # flagged, wrong
    ])
    y = np.array([0, 1, 0, 2])
    out = confidence_metrics(probs, y)

    assert out["threshold"] == C.LOW_CONFIDENCE
    assert out["flagged_fraction"] == 0.5
    assert out["n_confident"] == 2
    assert out["n_flagged"] == 2
    assert out["accuracy_when_confident"] == 0.5
    assert out["accuracy_when_flagged"] == 0.5


def test_confidence_reports_none_rather_than_nan_for_an_empty_side():
    """Nothing flagged must not produce a NaN that lands in metrics.json."""
    probs = _probs([[0.9, 0.1, 0.0, 0.0, 0.0, 0.0]])
    out = confidence_metrics(probs, np.array([0]))
    assert out["flagged_fraction"] == 0.0
    assert out["n_flagged"] == 0
    assert out["accuracy_when_confident"] == 1.0
    assert out["accuracy_when_flagged"] is None


def test_confidence_carries_the_denominator_for_a_near_empty_side():
    """Under heavy degradation almost everything is flagged, and the surviving
    accuracy is computed from a clip or two — useless without its n."""
    probs = _probs([
        [0.95, 0.05, 0.0, 0.0, 0.0, 0.0],   # the one clip still shown
        [0.3, 0.25, 0.2, 0.15, 0.1, 0.0],
        [0.3, 0.25, 0.2, 0.15, 0.1, 0.0],
        [0.3, 0.25, 0.2, 0.15, 0.1, 0.0],
    ])
    out = confidence_metrics(probs, np.array([1, 0, 0, 0]))
    assert out["accuracy_when_confident"] == 0.0   # looks damning ...
    assert out["n_confident"] == 1                 # ... on a single clip


def test_a_prediction_exactly_on_the_line_counts_as_confident():
    """The threshold is inclusive — the same comparison predict() makes."""
    p = C.LOW_CONFIDENCE
    probs = _probs([[p, 1.0 - p, 0.0, 0.0, 0.0, 0.0]])
    assert confidence_metrics(probs, np.array([0]))["flagged_fraction"] == 0.0
