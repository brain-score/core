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
def _mock_memory(baseline_rss, rss_after_probe, peak_rss=None):
    """Mock psutil RSS and resource peak RSS for check_memory.

    Args:
        baseline_rss: RSS before probe
        rss_after_probe: RSS after probe (sustained)
        peak_rss: peak RSS during probe (transient). If None, equals rss_after_probe.
    """
    if peak_rss is None:
        peak_rss = rss_after_probe
    call_count = [0]

    def mock_process(*args, **kwargs):
        mock = MagicMock()
        idx = min(call_count[0], 1)
        mock.memory_info.return_value.rss = [baseline_rss, rss_after_probe][idx]
        call_count[0] += 1
        return mock

    peak_count = [0]
    def mock_peak():
        peak_count[0] += 1
        if peak_count[0] <= 1:
            return baseline_rss  # before probe
        return peak_rss  # after probe

    with patch('psutil.Process', side_effect=mock_process), \
         patch('brainscore_core.memory._get_peak_rss', side_effect=mock_peak):
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
        with patch('brainscore_core.memory.get_available_memory', return_value=16_000_000_000), \
             _mock_memory(100, 200):
            check_memory(model, bench)  # should not raise

    def test_raises_when_insufficient_memory(self):
        # Ridge: baseline=500MB, extraction=200MB (probe delta)
        # metric(ridge, F=10K, S=1000, T=100):
        #   gram=min(1000,10000)^2*8=8MB, design=1000*10000*8=80MB,
        #   target=1000*100*8=0.8MB, centered=80MB, coef=10000*100*8=8MB
        #   total metric ≈ 177MB
        # total = 500 + 200 + 177 = 877MB. system=500+300=800MB. Fail.
        model = FakeModel(activation_nbytes=100_000, n_features=10000)
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=1000)
        with patch('brainscore_core.memory.get_available_memory', return_value=300_000_000), \
             _mock_memory(500_000_000, 700_000_000):
            with pytest.raises(MemoryError, match="Estimated peak"):
                check_memory(model, bench)

    def test_warns_at_high_utilization(self):
        # PLS: baseline=200MB, extraction=700MB, metric=small
        # total ≈ 900MB. system=200+900=1100MB. 82%. Warn.
        model = FakeModel(activation_nbytes=1000)
        bench = FakeBenchmark(identifier='test-pls', n_stimuli=10)
        with patch('brainscore_core.memory.get_available_memory', return_value=900_000_000), \
             _mock_memory(200_000_000, 900_000_000):
            with pytest.warns(ResourceWarning, match="OOM risk"):
                check_memory(model, bench)

    def test_graceful_when_probe_fails(self):
        model = FailingModel()
        bench = FakeBenchmark(n_stimuli=10)
        with patch('brainscore_core.memory.get_available_memory', return_value=16_000_000_000):
            check_memory(model, bench)  # should not raise

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
        # Ridge: baseline=500MB, extraction=200MB, metric≈177MB
        # total=877MB > system=800MB. Fail.
        model = FakeModel(identifier='big-vit', activation_nbytes=100_000, n_features=10000)
        bench = FakeBenchmark(identifier='MajajHong2015-ridge', n_stimuli=1000)
        with patch('brainscore_core.memory.get_available_memory', return_value=300_000_000), \
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
        with patch('brainscore_core.memory.get_available_memory', return_value=30_000_000_000), \
             _mock_memory(2_000_000_000, 2_500_000_000):
            check_memory(model, bench)  # should pass — dual Gram is tiny

    def test_extraction_peak_detected(self):
        # Extraction peaks at 3 GB (transient) but settles to 1.5 GB.
        # Metric adds 0.1 GB on top of settled RSS.
        # Peak = max(3 GB extraction, 1.5 + 0.1 GB metric) = 3 GB.
        # System = 4 GB. Should pass (3 < 4).
        model = FakeModel()
        bench = FakeBenchmark(identifier='test-pls', n_stimuli=10)
        with patch('brainscore_core.memory.get_available_memory', return_value=3_500_000_000), \
             _mock_memory(500_000_000, 1_500_000_000, peak_rss=3_000_000_000):
            check_memory(model, bench)  # passes — peak 3 GB < system 4 GB

        # Same but system only 2.4 GB. Peak 3 GB > system 2.9 GB. Fail.
        with patch('brainscore_core.memory.get_available_memory', return_value=2_400_000_000), \
             _mock_memory(500_000_000, 1_500_000_000, peak_rss=3_000_000_000):
            with pytest.raises(MemoryError):
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
        with patch('brainscore_core.memory.get_available_memory', return_value=900_000_000), \
             _mock_memory(100_000_000, 500_000_000):
            check_memory(model, bench_pls)  # passes — PLS metric is small

        with patch('brainscore_core.memory.get_available_memory', return_value=400_000_000), \
             _mock_memory(100_000_000, 500_000_000):
            with pytest.raises(MemoryError):
                check_memory(model, bench_cv_big)  # fails — RidgeCV metric is huge
