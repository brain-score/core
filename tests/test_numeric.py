"""Tests for brainscore_core.metrics.numeric.per_unit_pearson (the canonical
per-unit correlation that metrics + benchmark scorers share)."""
import numpy as np

from brainscore_core.metrics import per_unit_pearson


def test_matches_numpy_corrcoef_per_column():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((50, 6))
    B = 0.5 * A + 0.3 * rng.standard_normal((50, 6))
    r = per_unit_pearson(A, B)
    ref = np.array([np.corrcoef(A[:, j], B[:, j])[0, 1] for j in range(6)])
    assert np.allclose(r, ref)


def test_identical_inputs_give_one():
    A = np.random.default_rng(1).standard_normal((30, 4))
    assert np.allclose(per_unit_pearson(A, A), 1.0)


def test_zero_variance_column_is_nan():
    A = np.ones((10, 3))
    A[:, 0] = np.arange(10)                       # only column 0 varies
    B = np.random.default_rng(2).standard_normal((10, 3))
    r = per_unit_pearson(A, B)
    assert not np.isnan(r[0])
    assert np.isnan(r[1]) and np.isnan(r[2])      # constant columns → NaN
