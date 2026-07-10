"""Tests for ExecutionPlan — the benchmark memory-execution declaration."""
import pytest

from brainscore_core.execution_plan import ExecutionPlan


class TestDefaults:
    def test_metric_counts_default_to_extraction(self):
        p = ExecutionPlan(n_extraction_presentations=500, feature_width=1000)
        # metric defaults mirror extraction / raw width when not separately declared
        assert p.resolved_metric_observations == 500
        assert p.resolved_metric_feature_width(1000) == 1000  # falls back to raw

    def test_metric_counts_can_differ(self):
        p = ExecutionPlan(n_extraction_presentations=500, feature_width=38400,
                          metric_observations=100, metric_feature_width=1000)
        assert p.resolved_metric_observations == 100          # aggregated before metric
        assert p.resolved_metric_feature_width(38400) == 1000  # compressed before metric

    def test_feature_width_optional_falls_back_to_probed_raw(self):
        # feature_width omitted -> the metric width resolves against the probed raw
        p = ExecutionPlan(n_extraction_presentations=100)
        assert p.feature_width is None
        assert p.resolved_metric_feature_width(768) == 768

    def test_defaults(self):
        p = ExecutionPlan(n_extraction_presentations=1, feature_width=1)
        assert p.activation_dtype_bytes == 4
        assert p.runs_ceiling_metric is True
        assert p.metric_category is None
        assert p.recording_target is None
        assert p.probe_stimuli is None


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

    def test_non_bool_ceiling_flag_rejected(self):
        with pytest.raises(ValueError, match="runs_ceiling_metric must be a bool"):
            ExecutionPlan(n_extraction_presentations=1, feature_width=1,
                          runs_ceiling_metric=1)

    def test_bool_rejected_for_int_fields(self):
        # operator.index(True) == 1 would slip True through as a count — reject it.
        with pytest.raises(ValueError, match="must be an integer"):
            ExecutionPlan(n_extraction_presentations=True)

    def test_bad_metric_category_rejected(self):
        with pytest.raises(ValueError, match="metric_category must be one of"):
            ExecutionPlan(n_extraction_presentations=10, metric_category='banded')

    def test_numpy_int_accepted(self):
        np = pytest.importorskip('numpy')
        p = ExecutionPlan(n_extraction_presentations=np.int64(10),
                          feature_width=np.int64(20))
        assert p.n_extraction_presentations == 10
        assert p.feature_width == 20

    def test_frozen_after_construction(self):
        # Validation runs once at construction; the plan must be immutable so a
        # consumer can't be handed a mutated (invalid) plan.
        import dataclasses
        p = ExecutionPlan(n_extraction_presentations=1, feature_width=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.n_extraction_presentations = -1_000_000_000
