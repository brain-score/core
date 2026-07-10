"""Tests for ExecutionPlan — the benchmark memory-execution declaration."""
import pytest

from brainscore_core.execution_plan import ExecutionPlan


class TestDefaults:
    def test_metric_counts_default_to_extraction(self):
        p = ExecutionPlan(n_extraction_presentations=500, feature_width=1000)
        # metric defaults mirror extraction when not separately declared
        assert p.resolved_metric_observations == 500
        assert p.resolved_metric_feature_width == 1000

    def test_metric_counts_can_differ(self):
        p = ExecutionPlan(n_extraction_presentations=500, feature_width=38400,
                          metric_observations=100, metric_feature_width=1000)
        assert p.resolved_metric_observations == 100      # aggregated before metric
        assert p.resolved_metric_feature_width == 1000    # compressed before metric

    def test_dtype_and_device_defaults(self):
        p = ExecutionPlan(n_extraction_presentations=1, feature_width=1)
        assert p.activation_dtype_bytes == 4
        assert p.extraction_on_device is False


class TestValidation:
    @pytest.mark.parametrize("kwargs", [
        {'n_extraction_presentations': 0, 'feature_width': 10},
        {'n_extraction_presentations': -5, 'feature_width': 10},
        {'n_extraction_presentations': 10, 'feature_width': 0},
        {'n_extraction_presentations': 1.9, 'feature_width': 10},   # float rejected
        {'n_extraction_presentations': 10, 'feature_width': 10, 'metric_observations': 0},
        {'n_extraction_presentations': 10, 'feature_width': 10, 'metric_feature_width': -1},
        {'n_extraction_presentations': 10, 'feature_width': 10, 'activation_dtype_bytes': 0},
    ])
    def test_bad_values_rejected(self, kwargs):
        with pytest.raises(ValueError):
            ExecutionPlan(**kwargs)

    def test_non_bool_device_rejected(self):
        with pytest.raises(ValueError, match="extraction_on_device must be a bool"):
            ExecutionPlan(n_extraction_presentations=1, feature_width=1,
                          extraction_on_device=1)

    def test_numpy_int_accepted(self):
        np = pytest.importorskip('numpy')
        p = ExecutionPlan(n_extraction_presentations=np.int64(10),
                          feature_width=np.int64(20))
        assert p.n_extraction_presentations == 10
        assert p.feature_width == 20
