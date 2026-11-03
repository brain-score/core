import numpy as np
import pytest

from brainscore_core.benchmarks import ceil_score
from brainscore_core.metrics import Score


def test_scalar_score_is_divided_by_ceiling():
    raw, ceiling = Score(.4), Score(.8)
    raw.attrs['error'] = .05
    ceiled = ceil_score(raw, ceiling)
    assert float(ceiled) == pytest.approx(.5)
    assert ceiled.attrs['error'] == .05
    assert ceiled.attrs[Score.RAW_VALUES_KEY] is raw
    assert ceiled.attrs['ceiling'] is ceiling


def test_aggregation_score_keeps_center_and_error():
    raw = Score([.4, .1], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
    ceiling = Score([.8, .0], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
    ceiled = ceil_score(raw, ceiling)
    np.testing.assert_allclose(ceiled.values, [.5, .1])


def test_scalar_score_with_center_error_ceiling_uses_center():
    ceiling = Score([.8, .1], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
    ceiled = ceil_score(Score(.4), ceiling)
    assert ceiled.dims == ()
    assert float(ceiled) == pytest.approx(.5)
