import numpy as np
import pytest

from brainscore_core import na_score, score_or_na
from brainscore_core.compatibility import CompatibilityError, check_compatibility
from brainscore_core.metrics import Score


class _Model:
    identifier = "text-model"
    available_modalities = {"text"}
    required_modalities = set()
    region_layer_map = {}


class _Benchmark:
    identifier = "vision-benchmark"
    required_modalities = {"vision"}


def test_na_score_is_nan_and_distinguishable_from_zero():
    score = na_score("missing vision", error_type="CompatibilityError")
    zero = Score(0.0)

    assert np.isnan(score.item())
    assert score.attrs["status"] == "N/A"
    assert score.attrs["reason"] == "missing vision"
    assert score.attrs["error_type"] == "CompatibilityError"
    assert zero.item() == 0.0
    assert zero.attrs.get("status") != "N/A"


def test_score_or_na_converts_incompatible_pair_before_compute():
    calls = []

    def incompatible_score():
        check_compatibility(_Model(), _Benchmark())
        calls.append("computed")
        return Score(1.0)

    score = score_or_na(incompatible_score)

    assert calls == []
    assert np.isnan(score.item())
    assert score.attrs["status"] == "N/A"
    assert "vision" in score.attrs["reason"]


def test_score_or_na_leaves_compatible_score_unchanged():
    expected = Score(0.0)

    def compatible_score():
        return expected

    assert score_or_na(compatible_score) is expected
    assert expected.item() == 0.0
    assert expected.attrs.get("status") != "N/A"


def test_score_or_na_does_not_hide_runtime_failures():
    def failed_score():
        raise RuntimeError("model crashed")

    with pytest.raises(RuntimeError, match="model crashed"):
        score_or_na(failed_score)


def test_existing_compatibility_check_still_raises():
    with pytest.raises(CompatibilityError):
        check_compatibility(_Model(), _Benchmark())
