import pytest
import warnings
from typing import Dict, Set
from unittest.mock import patch, MagicMock

from brainscore_core.memory import (
    MemoryError,
    get_available_memory,
    get_peak_memory,
    reset_peak_memory,
    estimate_metric_memory,
    check_memory,
    _detect_metric_category,
    _get_n_targets,
    _get_n_stimuli,
    _get_n_alphas,
    SAFETY_FACTORS,
)
from brainscore_core.model_interface import UnifiedModel


# ── Helpers ──────────────────────────────────────────────────────────

class FakeResult:
    """Mimics a DataAssembly with nbytes and shape."""
    def __init__(self, nbytes=8000, n_features=1000):
        self.nbytes = nbytes
        self.shape = (1, n_features)


class FakeModel(UnifiedModel):

    def __init__(self, identifier='test-model', modalities=None,
                 region_layer_map=None, process_result=None,
                 activation_nbytes=8000, n_features=1000):
        self._id = identifier
        self._modalities = modalities or {'vision'}
        self._rlm = region_layer_map or {}
        self._process_result = process_result or FakeResult(nbytes=activation_nbytes, n_features=n_features)

    @property
    def identifier(self) -> str:
        return self._id

    @property
    def region_layer_map(self) -> Dict[str, str]:
        return self._rlm

    @property
    def supported_modalities(self) -> Set[str]:
        return self._modalities

    def process(self, stimuli):
        return self._process_result


class FailingModel(FakeModel):
    def process(self, stimuli):
        raise RuntimeError("model crashed")


class CountingFailingModel(FakeModel):
    """Counts process() calls (all raise) so a test can prove the probe was
    skipped, not merely that it failed."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.process_calls = 0

    def process(self, stimuli):
        self.process_calls += 1
        raise RuntimeError("model crashed")


class MultiRegionProbeModel(FakeModel):
    """A layer-aware candidate that only accepts process() AFTER start_recording
    (configured inside the benchmark's __call__, no .region attr). Records ONE
    region at a time; the width scales with how many regions were recorded (so a
    naive record-all fallback would inflate it), and an audio region 'A1' raises
    (a video wrapper can't serve it) — exercising cross-modality recovery."""
    def __init__(self, per_region_features, region_layer_map, **k):
        super().__init__(region_layer_map=region_layer_map, **k)
        self._recorded = None
        self._per = per_region_features

    def start_recording(self, target, time_bins=None, recording_type=None):
        self._recorded = [target] if isinstance(target, str) else list(target)

    def process(self, stimuli):
        if not self._recorded:
            raise RuntimeError("recording not started")
        if 'A1' in self._recorded:
            raise RuntimeError("audio layer not available on this wrapper")
        return type('R', (), {'shape': (1, self._per * len(self._recorded))})()


class FakeStimulusSet:
    """Minimal stimulus set with iloc support."""

    def __init__(self, n=10):
        self._n = n

    def __len__(self):
        return self._n

    @property
    def iloc(self):
        return self

    def __getitem__(self, key):
        return FakeStimulusSet(n=1)


class FakeAssembly:
    """Minimal assembly with sizes dict."""
    def __init__(self, n_neuroids=100, n_stimuli=10):
        self.sizes = {'neuroid': n_neuroids, 'presentation': n_stimuli}
        self.stimulus_set = FakeStimulusSet(n=n_stimuli)


class FakeBenchmark:

    def __init__(self, identifier='test-bench', n_stimuli=100,
                 n_targets=None, expected_feature_dim=None,
                 assembly=None):
        self.identifier = identifier
        self.stimulus_set = FakeStimulusSet(n=n_stimuli)
        if n_targets is not None:
            self.n_targets = n_targets
        if expected_feature_dim is not None:
            self.expected_feature_dim = expected_feature_dim
        if assembly is not None:
            self._assembly = assembly


from contextlib import contextmanager

@contextmanager
def _mock_memory(baseline_rss, *_ignored, **_kw):
    """Mock the process's baseline RSS for check_memory.

    The plan-based check reads baseline RSS once (no probe-RSS extrapolation), so
    only the first arg matters; trailing args are accepted for call-site compat.
    """
    def mock_process(*args, **kwargs):
        mock = MagicMock()
        mock.memory_info.return_value.rss = baseline_rss
        return mock

    with patch('psutil.Process', side_effect=mock_process):
        yield


# ── get_available_memory ─────────────────────────────────────────────

class TestGetAvailableMemory:

    def test_returns_positive_int(self):
        mem = get_available_memory()
        assert isinstance(mem, int)
        assert mem > 0

    def test_falls_back_to_psutil_without_torch(self):
        with patch.dict('sys.modules', {'torch': None}):
            mem = get_available_memory()
            assert mem > 0


# ── get_peak_memory ──────────────────────────────────────────────────

class TestGetPeakMemory:

    def test_returns_nonnegative_int(self):
        peak = get_peak_memory()
        assert isinstance(peak, int)
        assert peak >= 0


# ── reset_peak_memory ────────────────────────────────────────────────

class TestResetPeakMemory:

    def test_does_not_error(self):
        reset_peak_memory()  # should not raise


# ── _detect_metric_category ──────────────────────────────────────────

class TestDetectMetricCategory:

    def test_pls_from_identifier(self):
        bench = FakeBenchmark(identifier='MajajHong2015.IT-pls')
        assert _detect_metric_category(bench) == 'pls'

    def test_ridge_from_identifier(self):
        bench = FakeBenchmark(identifier='Allen2022_fmri_surface.IT-ridge')
        assert _detect_metric_category(bench) == 'ridge'

    def test_ridgecv_from_identifier(self):
        bench = FakeBenchmark(identifier='Papale2025.IT-ridgecv')
        assert _detect_metric_category(bench) == 'ridgecv'

    def test_ridgecv_preferred_over_ridge(self):
        bench = FakeBenchmark(identifier='something-ridgecv-split')
        assert _detect_metric_category(bench) == 'ridgecv'

    def test_rsa_from_rdm(self):
        bench = FakeBenchmark(identifier='Allen2022_fmri_surface.IT-rdm')
        assert _detect_metric_category(bench) == 'rsa'

    def test_behavioral_no_assembly(self):
        bench = FakeBenchmark(identifier='Rajalingham2018-i2n')
        assert _detect_metric_category(bench) == 'behavioral'

    def test_neural_with_assembly_defaults_to_pls(self):
        bench = FakeBenchmark(identifier='SomeBench',
                              assembly=FakeAssembly())
        assert _detect_metric_category(bench) == 'pls'

    def test_no_identifier(self):
        bench = MagicMock(spec=[])
        assert _detect_metric_category(bench) == 'behavioral'


class TestGetNTargets:

    def test_reads_from_assembly(self):
        bench = FakeBenchmark(assembly=FakeAssembly(n_neuroids=3000))
        assert _get_n_targets(bench) == 3000

    def test_reads_from_train_assembly(self):
        bench = FakeBenchmark()
        bench.train_assembly = FakeAssembly(n_neuroids=5000)
        assert _get_n_targets(bench) == 5000

    def test_falls_back_to_n_targets_attr(self):
        bench = FakeBenchmark(n_targets=200)
        assert _get_n_targets(bench) == 200

    def test_falls_back_to_default(self):
        bench = FakeBenchmark()
        assert _get_n_targets(bench) == 100


class TestGetNStimuli:

    def test_from_stimulus_set(self):
        bench = FakeBenchmark(n_stimuli=500)
        assert _get_n_stimuli(bench) == 500

    def test_from_assembly(self):
        bench = MagicMock(spec=['identifier', '_assembly'])
        bench.identifier = 'test'
        bench.stimulus_set = None
        bench._assembly = FakeAssembly(n_stimuli=300)
        assert _get_n_stimuli(bench) == 300


# ── estimate_metric_memory ───────────────────────────────────────────

class TestEstimateMetricMemory:

    def test_behavioral_returns_zero(self):
        bench = FakeBenchmark(identifier='Ferguson2024-value_delta', n_stimuli=50)
        assert estimate_metric_memory(bench) == 0

    def test_pls_includes_components(self):
        bench = FakeBenchmark(identifier='MajajHong2015.IT-pls', n_stimuli=100,
                              assembly=FakeAssembly(n_neuroids=600))
        mem = estimate_metric_memory(bench)
        assert mem > 0
        bench_cv = FakeBenchmark(identifier='test-ridgecv', n_stimuli=100,
                                 assembly=FakeAssembly(n_neuroids=600))
        assert mem < estimate_metric_memory(bench_cv)

    def test_ridgecv_scales_with_alphas(self):
        bench = FakeBenchmark(identifier='Papale2025.IT-ridgecv', n_stimuli=100,
                              assembly=FakeAssembly(n_neuroids=800))
        mem = estimate_metric_memory(bench)
        assert mem > 36_000_000
        bench_pls = FakeBenchmark(identifier='test-pls', n_stimuli=100,
                                  assembly=FakeAssembly(n_neuroids=800))
        assert mem > estimate_metric_memory(bench_pls) * 5

    def test_rsa_scales_quadratically(self):
        bench_small = FakeBenchmark(identifier='test-rdm', n_stimuli=50)
        bench_large = FakeBenchmark(identifier='test-rdm', n_stimuli=500)
        mem_small = estimate_metric_memory(bench_small)
        mem_large = estimate_metric_memory(bench_large)
        assert mem_large / mem_small == pytest.approx(100, rel=0.01)

    def test_zero_stimuli_pls(self):
        bench = FakeBenchmark(identifier='test-pls', n_stimuli=0)
        assert estimate_metric_memory(bench) == 0

    def test_ridge_uses_dual_gram(self):
        # When n_stimuli < n_features, Gram is (S, S) not (F, F)
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=100,
                              assembly=FakeAssembly(n_neuroids=200))
        mem_small_f = estimate_metric_memory(bench, n_features=100)
        mem_large_f = estimate_metric_memory(bench, n_features=100_000)
        # With 100 stimuli and 100K features: dual Gram is (100, 100),
        # same as (100, 100) for 100 features. Gram cost should be similar.
        # But design matrix scales with n_features.
        assert mem_large_f > mem_small_f  # design matrix is larger
        # Gram should NOT scale quadratically with features when S < F
        assert mem_large_f < mem_small_f * 1000  # not F^2 scaling


# ── check_memory ─────────────────────────────────────────────────────

class TestCheckMemory:

    def test_passes_when_plenty_of_memory(self):
        model = FakeModel()
        bench = FakeBenchmark(n_stimuli=10)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=16_000_000_000), \
             _mock_memory(100, 200):
            check_memory(model, bench)  # should not raise

    def test_raises_when_insufficient_memory(self):
        # Ridge plan, baseline=500MB, F=20K, S=1000:
        #   held = 1000*20000*4 = 80MB; extraction = 120MB
        #   metric: design=1000*20000*8=160MB, centered=160MB, gram=8MB,
        #           coef=20000*100*8=16MB -> ~345MB; *1.2 -> ~417MB
        #   peak_metric = 500 + 80 + 417 = 997MB. system = 500+300 = 800MB. Fail.
        model = FakeModel(n_features=20000)
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=1000)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=300_000_000), \
             _mock_memory(500_000_000, 700_000_000):
            with pytest.raises(MemoryError, match="Estimated peak"):
                check_memory(model, bench)

    def test_warns_at_high_utilization(self):
        # Ridge plan, baseline=200MB, F=100K, S=1000:
        #   held=400MB; metric design+centered≈1600MB, coef=80MB -> ~1688MB;
        #   *1.2 -> ~2028MB. peak_metric = 200+400+2028 = ~2628MB.
        #   system = 200+2900 = 3100MB -> ~85% utilization. Warn, no raise.
        model = FakeModel(n_features=100000)
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=1000)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=2_900_000_000), \
             _mock_memory(200_000_000, 200_000_000):
            with pytest.warns(ResourceWarning, match="OOM risk"):
                check_memory(model, bench)

    def test_graceful_when_probe_fails(self):
        model = FailingModel()
        bench = FakeBenchmark(n_stimuli=10)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=16_000_000_000):
            check_memory(model, bench)  # should not raise

    def test_probe_failure_warns_loudly(self, caplog):
        # If no region yields a readable feature dim, the check is skipped — but
        # LOUDLY (WARN + traceback), never a silent INFO skip. The warning states
        # the guard is OFF for this run (not that the skip is "safe").
        import logging
        model = FailingModel()
        bench = FakeBenchmark(n_stimuli=10)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=16_000_000_000):
            with caplog.at_level(logging.WARNING, logger='brainscore_core.memory'):
                check_memory(model, bench)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "probe failure should emit a WARNING, not a silent INFO skip"
        assert 'could NOT run' in warnings[-1].getMessage()
        # WARN+traceback, not WARN+message: the record must carry exc_info so the
        # log shows WHERE the probe failed, not just the exception type/message.
        assert warnings[-1].exc_info is not None
        assert warnings[-1].exc_info[0] is RuntimeError  # FailingModel.process raises

    def test_feature_dim_argument_skips_the_probe(self):
        # The fully-declarative path: with feature_dim given, process() is NEVER
        # called (asserted via the spy) and the declared dim drives the estimate.
        model = CountingFailingModel()
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=1000)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=300_000_000), \
             _mock_memory(500_000_000):
            with pytest.raises(MemoryError, match="feature_dim argument"):
                check_memory(model, bench, feature_dim=100000)
        assert model.process_calls == 0  # probe never ran

    def test_expected_feature_dim_declaration_skips_the_probe(self):
        # Same, via benchmark.expected_feature_dim: process() is never called.
        model = CountingFailingModel()
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=1000,
                              expected_feature_dim=100000)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=300_000_000), \
             _mock_memory(500_000_000):
            with pytest.raises(MemoryError):
                check_memory(model, bench)
        assert model.process_calls == 0  # probe never ran

    def test_negative_feature_dim_rejected(self):
        model = FakeModel()
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=10)
        with pytest.raises(ValueError, match="positive integer"):
            check_memory(model, bench, feature_dim=-1)

    def test_probe_records_one_region_not_all_and_recovers_from_bad_region(self):
        # Finding 2: the probe must record a SINGLE region (not sum every region's
        # layers) and must skip a region whose modality it can't serve. Model has
        # {'A1','IT'}: A1 (audio) raises, IT works. Correct behaviour -> width is
        # ONE region (50000), not summed (100000), proving both properties.
        model = MultiRegionProbeModel(per_region_features=50000,
                                      region_layer_map={'A1': 'a', 'IT': 'i'})
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=1000)  # no .region
        with patch('brainscore_core.memory.get_host_available_memory', return_value=300_000_000), \
             _mock_memory(500_000_000):
            with pytest.raises(MemoryError, match=r"n_features: 50000 ") as exc:
                check_memory(model, bench)
        assert "100000" not in str(exc.value)  # NOT the summed 2-region width

    def test_expected_n_presentations_drives_extraction(self):
        # Finding 1a: a row-expanding benchmark declares the true count; extraction
        # uses it. Raw len=10 would pass; declared 500K forces a raise.
        model = FakeModel(n_features=2000)
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=10)
        bench.expected_n_presentations = 500_000
        with patch('brainscore_core.memory.get_host_available_memory', return_value=1_000_000_000), \
             _mock_memory(200_000_000):
            with pytest.raises(MemoryError, match=r"n_presentations: 500000 \[declared\]"):
                check_memory(model, bench)

    def test_expected_n_presentations_sizes_the_metric_too(self):
        # Finding 1 (the real bug): the METRIC term must fit over the declared
        # count, not raw len. RSA metric is S^2, features are irrelevant (=1), so
        # extraction is tiny; only a metric that uses the declared 500K raises.
        model = FakeModel(n_features=1)
        bench = FakeBenchmark(identifier='test-rsa', n_stimuli=10)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=100_000_000_000), \
             _mock_memory(200_000_000):
            check_memory(model, bench)  # raw S=10 -> RSA metric ~1.6KB -> passes
        bench.expected_n_presentations = 500_000  # RSA metric ~4000 GB -> must raise
        with patch('brainscore_core.memory.get_host_available_memory', return_value=100_000_000_000), \
             _mock_memory(200_000_000):
            with pytest.raises(MemoryError):
                check_memory(model, bench)

    def test_non_integer_declarations_rejected(self):
        # Finding 5: floats truncate silently and zero/negative slip through
        # truthiness — reject them explicitly.
        model = FakeModel()
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=10)
        with pytest.raises(ValueError, match="must be an integer"):
            check_memory(model, bench, feature_dim=1.9)
        bench0 = FakeBenchmark(identifier='test-ridge', n_stimuli=10,
                               expected_feature_dim=0)
        with pytest.raises(ValueError, match="positive integer"):
            check_memory(model, bench0)
        benchneg = FakeBenchmark(identifier='test-ridge', n_stimuli=10)
        benchneg.expected_n_presentations = -5
        with pytest.raises(ValueError, match="positive integer"):
            check_memory(model, benchneg)

    def test_undeclared_pass_is_flagged_approximate(self, caplog):
        # Finding 1/3/4: a fit for an un-declared benchmark must NOT read as an
        # all-clear — it is logged as APPROXIMATE (plan not modelled).
        import logging
        model = FakeModel(n_features=1000)
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=10)  # no declarations
        with patch('brainscore_core.memory.get_host_available_memory', return_value=16_000_000_000), \
             _mock_memory(200_000_000):
            with caplog.at_level(logging.INFO, logger='brainscore_core.memory'):
                check_memory(model, bench)
        assert any('APPROXIMATE' in r.getMessage() for r in caplog.records)

    def test_declared_plan_is_still_flagged_approximate(self, caplog):
        # Finding 3 regression guard: even when BOTH cardinality and feature width
        # are declared, temporal multiplicity / metric-observation count / GPU stay
        # unmodelled, so APPROXIMATE must still appear. Fails against the parent,
        # where declaring both suppressed the flag.
        import logging
        model = FakeModel(n_features=1000)
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=10,
                              expected_feature_dim=1000)
        bench.expected_n_presentations = 100
        with patch('brainscore_core.memory.get_host_available_memory', return_value=16_000_000_000), \
             _mock_memory(200_000_000):
            with caplog.at_level(logging.INFO, logger='brainscore_core.memory'):
                check_memory(model, bench)
        assert any('APPROXIMATE' in r.getMessage() for r in caplog.records)

    # ── ExecutionPlan (reliable) path ────────────────────────────────

    def test_execution_plan_skips_the_probe(self):
        # A declared plan means no probe: CountingFailingModel would raise if
        # probed, but the plan drives the estimate and process() is never called.
        from brainscore_core.execution_plan import ExecutionPlan
        model = CountingFailingModel()
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=10)
        bench.execution_plan = ExecutionPlan(n_extraction_presentations=1000,
                                             feature_width=100000)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=300_000_000), \
             _mock_memory(500_000_000):
            with pytest.raises(MemoryError, match="RELIABLE for host RAM"):
                check_memory(model, bench)
        assert model.process_calls == 0

    def test_execution_plan_result_is_not_flagged_approximate(self, caplog):
        import logging
        from brainscore_core.execution_plan import ExecutionPlan
        model = FakeModel()
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=10)
        bench.execution_plan = ExecutionPlan(n_extraction_presentations=100,
                                             feature_width=1000)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=16_000_000_000), \
             _mock_memory(200_000_000):
            with caplog.at_level(logging.INFO, logger='brainscore_core.memory'):
                check_memory(model, bench)
        msgs = [r.getMessage() for r in caplog.records]
        assert any('RELIABLE' in m for m in msgs)
        assert not any('APPROXIMATE' in m for m in msgs)

    def test_metric_observations_size_the_metric_independently(self):
        # metric_observations >> extraction: RSA metric is S^2 over metric_obs, so
        # a tiny 10-row extraction but 500K-row metric must raise from the metric.
        from brainscore_core.execution_plan import ExecutionPlan
        model = FakeModel()
        bench = FakeBenchmark(identifier='test-rsa', n_stimuli=10)
        bench.execution_plan = ExecutionPlan(n_extraction_presentations=10,
                                             feature_width=1,
                                             metric_observations=500_000)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=50_000_000_000), \
             _mock_memory(200_000_000):
            with pytest.raises(MemoryError):
                check_memory(model, bench)

    def test_metric_feature_width_sizes_the_metric_independently(self):
        # feature_width tiny (extraction ~0) but metric_feature_width huge -> the
        # metric term must use the compressed... here INFLATED metric width, proving
        # it is independent of the extraction width.
        from brainscore_core.execution_plan import ExecutionPlan
        model = FakeModel()
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=10)
        bench.execution_plan = ExecutionPlan(n_extraction_presentations=1,
                                             feature_width=1,
                                             metric_observations=100,
                                             metric_feature_width=100000)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=100_000_000), \
             _mock_memory(200_000_000):
            with pytest.raises(MemoryError):
                check_memory(model, bench)
        # control: small metric width -> no raise (extraction is tiny)
        bench.execution_plan = ExecutionPlan(n_extraction_presentations=1,
                                             feature_width=1,
                                             metric_observations=100,
                                             metric_feature_width=1)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=100_000_000), \
             _mock_memory(200_000_000):
            check_memory(model, bench)  # passes

    def test_extraction_on_device_excludes_the_host_matrix(self):
        # A huge on-device activation matrix must NOT count against host RAM; only
        # the (small, aggregated) metric does. Same plan on host would raise.
        from brainscore_core.execution_plan import ExecutionPlan
        model = FakeModel()
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=10)
        host_plan = dict(n_extraction_presentations=1_000_000, feature_width=100000,
                         metric_observations=100, metric_feature_width=1000)
        # on host: held = 1e6*1e5*4 = 400 GB -> raises
        bench.execution_plan = ExecutionPlan(**host_plan, extraction_on_device=False)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=8_000_000_000), \
             _mock_memory(500_000_000):
            with pytest.raises(MemoryError):
                check_memory(model, bench)
        # on device: host holds only the small metric -> passes
        bench.execution_plan = ExecutionPlan(**host_plan, extraction_on_device=True)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=8_000_000_000), \
             _mock_memory(500_000_000):
            check_memory(model, bench)  # passes

    def test_execution_plan_as_method_is_honored(self):
        from brainscore_core.execution_plan import ExecutionPlan
        model = CountingFailingModel()
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=10)
        bench.execution_plan = lambda: ExecutionPlan(
            n_extraction_presentations=1000, feature_width=100000)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=300_000_000), \
             _mock_memory(500_000_000):
            with pytest.raises(MemoryError, match="RELIABLE"):
                check_memory(model, bench)
        assert model.process_calls == 0

    def test_non_plan_execution_plan_rejected(self):
        model = FakeModel()
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=10)
        bench.execution_plan = {'not': 'a plan'}
        with pytest.raises(TypeError, match="must be an ExecutionPlan"):
            check_memory(model, bench)

    def test_shapeless_probe_result_warns_and_skips(self, caplog):
        import logging
        # process() returns a 1-D result -> feature width unreadable -> skip loudly.
        model = FakeModel(process_result=type('R', (), {'shape': (5,)})())
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=10)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=16_000_000_000), \
             _mock_memory(200_000_000):
            with caplog.at_level(logging.WARNING, logger='brainscore_core.memory'):
                check_memory(model, bench)  # no raise; skipped
        assert any('could NOT run' in r.getMessage() for r in caplog.records)

    def test_graceful_when_no_stimulus_set(self):
        model = FakeModel()
        bench = MagicMock(spec=['identifier'])
        bench.identifier = 'no-stimuli-bench'
        check_memory(model, bench)  # should not raise

    def test_graceful_when_empty_stimulus_set(self):
        model = FakeModel()
        bench = FakeBenchmark(n_stimuli=0)
        check_memory(model, bench)  # should not raise

    def test_error_message_includes_identifiers(self):
        # Ridge plan, F=20K, S=1000 (same as the raise test): ~997MB > 800MB. Fail.
        model = FakeModel(identifier='big-vit', n_features=20000)
        bench = FakeBenchmark(identifier='MajajHong2015-ridge', n_stimuli=1000)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=300_000_000), \
             _mock_memory(500_000_000, 700_000_000):
            with pytest.raises(MemoryError, match="big-vit") as exc_info:
                check_memory(model, bench)
            assert "MajajHong2015" in str(exc_info.value)

    def test_large_features_use_dual_gram(self):
        # Model with 100K features but only 500 stimuli.
        # Ridge dual Gram: min(500, 100K)^2 * 8 = 2 MB (not 80 GB!)
        # coef: 100K * 100 * 8 = 80 MB. Total metric ≈ 882 MB.
        # total = 2GB + 500MB + 882MB = 3.4 GB < 32 GB. Pass.
        model = FakeModel(activation_nbytes=400_000, n_features=100_000)
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=500)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=30_000_000_000), \
             _mock_memory(2_000_000_000, 2_500_000_000):
            check_memory(model, bench)  # should pass — dual Gram is tiny

    def test_extraction_is_the_bottleneck_when_metric_is_cheap(self):
        # Behavioral metric ≈ 0, so the extraction transient (1.5x held) is the
        # peak: F=100K, S=2000 -> held=800MB, extraction=1200MB.
        # baseline=500MB -> peak_extraction=1700MB > peak_metric=1300MB.
        model = FakeModel(n_features=100000)
        bench = FakeBenchmark(identifier='test-behavioral', n_stimuli=2000)
        # system = 500+3000 = 3500MB. 1700 < 3500 -> pass.
        with patch('brainscore_core.memory.get_host_available_memory', return_value=3_000_000_000), \
             _mock_memory(500_000_000, 500_000_000):
            check_memory(model, bench)
        # system = 500+1000 = 1500MB. 1700 > 1500 -> fail (extraction-bound).
        with patch('brainscore_core.memory.get_host_available_memory', return_value=1_000_000_000), \
             _mock_memory(500_000_000, 500_000_000):
            with pytest.raises(MemoryError, match="extraction"):
                check_memory(model, bench)

    def test_ridgecv_metric_dominates(self):
        """RidgeCV with many alphas produces large metric estimate."""
        model = FakeModel(activation_nbytes=1000)
        bench_pls = FakeBenchmark(identifier='test-pls', n_stimuli=100)
        bench_cv = FakeBenchmark(identifier='test-ridgecv', n_stimuli=100)
        # Same extraction overhead (400MB), but RidgeCV metric >> PLS metric.
        # PLS metric ≈ small. RidgeCV metric with 115 alphas ≈ large.
        # baseline=100MB, extraction=400MB
        # PLS total ≈ 500 + small = fits in 1GB system. Pass.
        # RidgeCV total ≈ 500 + LOO(100*100*115*8=92MB) + design + ... > 1GB?
        # Actually with default n_targets=100: LOO = 92MB. Still fits.
        # Use n_targets=5000 to make it fail:
        bench_cv_big = FakeBenchmark(identifier='test-ridgecv', n_stimuli=100,
                                     assembly=FakeAssembly(n_neuroids=5000))
        # LOO = 100 * 5000 * 115 * 8 = 460 MB. + design + centered + gram = ~500MB
        # total = 100 + 400 + 960 = 1460 MB > system 1000MB. Fail.
        with patch('brainscore_core.memory.get_host_available_memory', return_value=900_000_000), \
             _mock_memory(100_000_000, 500_000_000):
            check_memory(model, bench_pls)  # passes — PLS metric is small

        with patch('brainscore_core.memory.get_host_available_memory', return_value=400_000_000), \
             _mock_memory(100_000_000, 500_000_000):
            with pytest.raises(MemoryError):
                check_memory(model, bench_cv_big)  # fails — RidgeCV metric is huge
