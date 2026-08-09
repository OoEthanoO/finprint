"""The prediction pipeline's graceful-degradation guarantee.

finprint.predict promises that the acoustic and call-type halves still work
without a trained checkpoint — species is simply reported as unavailable. That
path is easy to break silently (the model loader has to stay unpackable whether
or not a checkpoint exists), so it is pinned here.
"""
import io
from pathlib import Path

import numpy as np
import soundfile as sf

import finprint.config as C
import finprint.predict as P
from signals import tone


def _clip(tmp_path: Path) -> str:
    wav = tone(1200.0, 1.0)
    path = tmp_path / "clip.wav"
    sf.write(path, wav, C.SAMPLE_RATE)
    return str(path)


def test_loader_arity_is_stable_without_a_checkpoint(monkeypatch, tmp_path):
    """The no-checkpoint result must unpack like the loaded one."""
    monkeypatch.setattr(C, "CHECKPOINT", tmp_path / "absent.pt")
    P._load_model.cache_clear()
    try:
        model, labels, norm, dev = P._load_model()   # must not raise
        assert model is None and labels is None and norm is None and dev is None
    finally:
        P._load_model.cache_clear()


def test_predict_degrades_without_a_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "CHECKPOINT", tmp_path / "absent.pt")
    P._load_model.cache_clear()
    try:
        out = P.predict(_clip(tmp_path), with_spectrogram=False)
    finally:
        P._load_model.cache_clear()

    assert out["species"] is None
    assert out["model_available"] is False
    # the parts that do not need a model still have to be there
    assert out["call_type"]["label"]
    assert out["features"]["duration_s"] > 0


def test_taxonomy_covers_every_label_the_model_can_emit(tmp_path):
    """A species with no group would silently drop the group from the response."""
    import json

    import pytest

    from finprint.taxonomy import group_of

    if not C.LABELMAP.exists():
        pytest.skip("no label map in this checkout")
    labels = json.loads(C.LABELMAP.read_text())
    unmapped = [n for n in labels.values() if group_of(n) is None]
    assert unmapped == [], f"species missing a taxonomic group: {unmapped}"


def test_group_never_contradicts_the_top_species(tmp_path):
    """The group is derived from the top species, so they must always agree."""
    import pytest

    from finprint.taxonomy import group_of

    if not C.CHECKPOINT.exists():
        pytest.skip("no trained checkpoint in this checkout")
    P._load_model.cache_clear()
    out = P.predict(_clip(tmp_path), with_spectrogram=False)
    assert out["group"]["label"] == group_of(out["species"][0]["species"])
    # the group's mass must be at least the top species' own probability
    assert out["group"]["confidence"] >= out["species"][0]["confidence"] - 1e-6
    assert 0.0 <= out["group"]["confidence"] <= 1.0


def test_low_confidence_flag_matches_the_measured_line(tmp_path):
    """The UI renders this flag verbatim, so it must be the threshold decision
    and not something the page has to re-derive from the top score."""
    import pytest

    if not C.CHECKPOINT.exists():
        pytest.skip("no trained checkpoint in this checkout")
    P._load_model.cache_clear()
    out = P.predict(_clip(tmp_path), with_spectrogram=False)
    assert out["low_confidence"] is (
        out["species"][0]["confidence"] < C.LOW_CONFIDENCE
    )


def test_low_confidence_is_none_without_a_checkpoint(monkeypatch, tmp_path):
    """No species means no claim to qualify — not a claim of high confidence."""
    monkeypatch.setattr(C, "CHECKPOINT", tmp_path / "absent.pt")
    P._load_model.cache_clear()
    try:
        out = P.predict(_clip(tmp_path), with_spectrogram=False)
    finally:
        P._load_model.cache_clear()
    assert out["low_confidence"] is None


def test_group_absent_without_a_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "CHECKPOINT", tmp_path / "absent.pt")
    P._load_model.cache_clear()
    try:
        out = P.predict(_clip(tmp_path), with_spectrogram=False)
    finally:
        P._load_model.cache_clear()
    assert out["group"] is None


def test_predict_reports_species_when_a_checkpoint_exists(tmp_path):
    if not C.CHECKPOINT.exists():
        import pytest
        pytest.skip("no trained checkpoint in this checkout")
    P._load_model.cache_clear()
    out = P.predict(_clip(tmp_path), with_spectrogram=False)
    assert out["model_available"] is True
    assert len(out["species"]) == 3
    assert 0.0 <= out["species"][0]["confidence"] <= 1.0
    # top-k must come back ordered
    confs = [s["confidence"] for s in out["species"]]
    assert confs == sorted(confs, reverse=True)
