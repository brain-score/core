import numpy as np
import pytest

import brainscore_core
from brainscore_core import STATUS_NA, na_score
from brainscore_core.compatibility import CompatibilityError, check_compatibility
from brainscore_core.metrics import Score


class _Model:
    # Compatibility reads declarations only (Contract v3 rule 2.1).
    identifier = "text-model"
    in_channels = {"text"}
    out_channels = {"behavior"}
    required_channels = set()
    output_inputs = {}
    traits = frozenset()


class _Benchmark:
    identifier = "neural-benchmark"
    required_input_channels = {"text"}
    requested_output_channels = {"neural:IT"}


def test_na_score_is_nan_and_distinguishable_from_zero():
    score = na_score(CompatibilityError("missing vision", channels={"vision:image"}))
    zero = Score(0.0)

    assert np.isnan(score.item())
    assert score.attrs["status"] == STATUS_NA == "N/A"
    assert score.attrs["reason"] == "missing vision"
    assert score.attrs["missing_channels"] == ["vision:image"]
    assert zero.item() == 0.0
    assert zero.attrs.get("status") != "N/A"


def test_preflight_error_names_the_missing_channels():
    with pytest.raises(CompatibilityError) as caught:
        check_compatibility(_Model(), _Benchmark())
    assert caught.value.channels == {"neural:IT"}
    score = na_score(caught.value)
    assert score.attrs["missing_channels"] == ["neural:IT"]
    assert "neural:IT" in score.attrs["reason"]


def test_an_error_without_channels_gives_an_empty_list():
    assert na_score(CompatibilityError("no part for the task")).attrs["missing_channels"] == []


def test_there_is_no_separate_conversion_helper():
    # score() returns the N/A result itself; no wrapper converts the error.
    assert not hasattr(brainscore_core, "score_or_na")
