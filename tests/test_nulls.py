"""Tests for the nulls validation framework. Each null must (a) be
deterministic given a seed, (b) return a copy (never mutate input), and
(c) destroy signal in the intended way — verified against a tiny synthetic
encoding problem where the true score is high and every null collapses it.
"""
import numpy as np
import pytest

from brainscore_core import nulls


def _ridge_r(X, Y, *, alpha=1.0):
    """Minimal closed-form ridge + Pearson r on a held-out half. Self-contained
    so the test has no sklearn dependency and the 'real' score is unambiguous."""
    n = X.shape[0]
    half = n // 2
    Xtr, Ytr = X[:half], Y[:half]
    Xte, Yte = X[half:], Y[half:]
    # center
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    Xtr, Xte = Xtr - xm, Xte - xm
    Ytr_c = Ytr - ym
    W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]), Xtr.T @ Ytr_c)
    pred = Xte @ W + ym
    # mean Pearson r over targets
    rs = []
    for j in range(Y.shape[1]):
        a, b = pred[:, j], Yte[:, j]
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        rs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(rs)) if rs else 0.0


@pytest.fixture
def encoding_problem():
    rng = np.random.RandomState(0)
    n, d, t = 200, 8, 5
    X = rng.randn(n, d)
    Wtrue = rng.randn(d, t)
    Y = X @ Wtrue + 0.1 * rng.randn(n, t)   # strong, learnable signal
    return X, Y


class TestShuffleRows:
    def test_deterministic_and_copy(self):
        X = np.arange(20).reshape(10, 2)
        a = nulls.shuffle_rows(X, seed=3)
        b = nulls.shuffle_rows(X, seed=3)
        assert np.array_equal(a, b)
        assert not np.array_equal(X, a)          # input untouched
        assert sorted(a[:, 0].tolist()) == sorted(X[:, 0].tolist())

    def test_collapses_encoding_score(self, encoding_problem):
        X, Y = encoding_problem
        real = _ridge_r(X, Y)
        null = _ridge_r(nulls.shuffle_rows(X, seed=1), Y)
        assert real > 0.8
        assert null < 0.2
        assert real - null > 0.5


class TestShiftFeatures:
    def test_circular_and_copy(self):
        X = np.arange(12).reshape(6, 2).astype(float)
        s = nulls.shift_features(X, 2)
        assert s.shape == X.shape
        assert np.allclose(s, np.roll(X, 2, axis=0))
        assert not np.shares_memory(s, X)

    def test_degrades_temporally_structured_signal(self):
        # Build a signal where Y[t] depends on X[t] -> shifting breaks it.
        rng = np.random.RandomState(2)
        n, d = 240, 6
        X = rng.randn(n, d)
        W = rng.randn(d, 4)
        Y = X @ W + 0.1 * rng.randn(n, 4)
        real = _ridge_r(X, Y)
        shifted = _ridge_r(nulls.shift_features(X, 7), Y)
        assert real > 0.8
        assert shifted < 0.3

    def test_curve_peaks_at_zero(self):
        rng = np.random.RandomState(4)
        n, d = 240, 6
        X = rng.randn(n, d)
        W = rng.randn(d, 4)
        Y = X @ W + 0.1 * rng.randn(n, 4)
        curve = nulls.temporal_shift_curve(lambda Xs: _ridge_r(Xs, Y), X,
                                           shifts=[-10, -3, 0, 3, 10])
        assert curve[0] == max(curve.values())   # best alignment at shift 0
        assert curve[0] > curve[10] + 0.3


class TestShuffleLabels:
    def test_permutes_and_copies(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        s = nulls.shuffle_labels(y, seed=5)
        assert sorted(s.tolist()) == sorted(y.tolist())
        assert s.tolist() == nulls.shuffle_labels(y, seed=5).tolist()


class TestRandomUnitSubset:
    def test_size_sorted_unique(self):
        idx = nulls.random_unit_subset(100, 10, seed=0)
        assert len(idx) == 10
        assert len(set(idx.tolist())) == 10
        assert list(idx) == sorted(idx)

    def test_too_large_raises(self):
        with pytest.raises(ValueError, match="exceeds"):
            nulls.random_unit_subset(5, 10)


class TestRandomAction:
    def test_shape_range_deterministic(self):
        a = nulls.random_action((7,), seed=1, low=-1, high=1)
        assert a.shape == (7,)
        assert a.min() >= -1 and a.max() <= 1
        assert np.array_equal(a, nulls.random_action((7,), seed=1))


class TestDescribe:
    def test_known_capability(self):
        got = nulls.describe('temporal_multimodal')
        assert 'shift_features' in got and 'shuffle_rows' in got

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown capability"):
            nulls.describe('telepathy')


class TestResolve:
    def test_every_capability_null_is_classified(self):
        # No orphan names: every null in every capability must be callable here,
        # model-level, or perturbation-level. (Catches the phantom-null bug.)
        known = nulls.CALLABLE_NULLS | nulls.MODEL_LEVEL_NULLS | nulls.PERTURBATION_NULLS
        for cap in nulls.NULLS_BY_CAPABILITY:
            for name in nulls.describe(cap):
                assert name in known, f"{name} in {cap} is unclassified"

    def test_callable_nulls_resolve_to_functions(self):
        for name in nulls.CALLABLE_NULLS:
            fn = nulls.resolve(name)
            assert callable(fn)

    def test_model_level_null_raises_clear_error(self):
        with pytest.raises(ValueError, match="MODEL-level null"):
            nulls.resolve('random_init_model')

    def test_perturbation_null_raises_clear_error(self):
        with pytest.raises(ValueError, match="PERTURBATION-level null"):
            nulls.resolve('opposite_sign')

    def test_callable_nulls_filter(self):
        cn = nulls.callable_nulls('behavioral')
        assert 'shuffle_labels' in cn
        assert 'chance_baseline' not in cn          # model-level, filtered out
