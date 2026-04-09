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
    """Mimics a DataAssembly with nbytes."""
    def __init__(self, nbytes=8000):
        self.nbytes = nbytes


class FakeModel(UnifiedModel):

    def __init__(self, identifier='test-model', modalities=None,
                 region_layer_map=None, process_result=None,
                 activation_nbytes=8000):
        self._id = identifier
        self._modalities = modalities or {'vision'}
        self._rlm = region_layer_map or {}
        self._process_result = process_result or FakeResult(nbytes=activation_nbytes)

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
        # Should include design + target + deflated + components
        assert mem > 0
        # PLS should be smaller than ridgecv for same dimensions
        bench_cv = FakeBenchmark(identifier='test-ridgecv', n_stimuli=100,
                                 assembly=FakeAssembly(n_neuroids=600))
        assert mem < estimate_metric_memory(bench_cv)

    def test_ridgecv_scales_with_alphas(self):
        bench = FakeBenchmark(identifier='Papale2025.IT-ridgecv', n_stimuli=100,
                              assembly=FakeAssembly(n_neuroids=800))
        mem = estimate_metric_memory(bench)
        # RidgeCV with 115 alphas: loo = 100 * 800 * 115 * 4 = 36.8 MB
        # Plus gram (4MB) + design (0.4MB) + target (0.32MB) ≈ 41.5 MB
        assert mem > 36_000_000
        # Verify it's much larger than PLS for same dimensions
        bench_pls = FakeBenchmark(identifier='test-pls', n_stimuli=100,
                                  assembly=FakeAssembly(n_neuroids=800))
        assert mem > estimate_metric_memory(bench_pls) * 5

    def test_rsa_scales_quadratically(self):
        bench_small = FakeBenchmark(identifier='test-rdm', n_stimuli=50)
        bench_large = FakeBenchmark(identifier='test-rdm', n_stimuli=500)
        mem_small = estimate_metric_memory(bench_small)
        mem_large = estimate_metric_memory(bench_large)
        # 500^2 / 50^2 = 100x ratio
        assert mem_large / mem_small == pytest.approx(100, rel=0.01)

    def test_zero_stimuli_pls(self):
        bench = FakeBenchmark(identifier='test-pls', n_stimuli=0)
        assert estimate_metric_memory(bench) == 0

    def test_ridge_includes_gram_matrix(self):
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=100,
                              assembly=FakeAssembly(n_neuroids=200))
        mem = estimate_metric_memory(bench)
        # gram = 1000 * 1000 * 4 = 4MB. Must be included.
        assert mem >= 4_000_000


# ── check_memory ─────────────────────────────────────────────────────

def _mock_rss(*rss_values):
    """Create a mock for psutil.Process().memory_info().rss that returns
    successive values on each call."""
    call_count = [0]

    def mock_process(*args, **kwargs):
        mock = MagicMock()
        idx = min(call_count[0], len(rss_values) - 1)
        mock.memory_info.return_value.rss = rss_values[idx]
        call_count[0] += 1
        return mock

    return patch('psutil.Process', side_effect=mock_process)


class TestCheckMemory:

    def test_passes_when_plenty_of_memory(self):
        model = FakeModel()
        bench = FakeBenchmark(n_stimuli=10)
        # RSS barely moves: 100 → 200 → 200 (second probe same)
        with patch('brainscore_core.memory.get_available_memory', return_value=16_000_000_000), \
             _mock_rss(100, 200, 200):
            check_memory(model, bench)  # should not raise

    def test_raises_when_insufficient_memory(self):
        # PLS benchmark: baseline=500MB, after probe=600MB, after 2nd=605MB
        # marginal=5MB/stim, forward=100MB, n_stimuli=1000
        # total = 500MB + (100MB + 5MB*1000 + metric) * 1.5 > 2GB. Available=1.5GB.
        model = FakeModel(activation_nbytes=100_000)
        bench = FakeBenchmark(identifier='test-pls', n_stimuli=1000)
        with patch('brainscore_core.memory.get_available_memory', return_value=1_500_000_000), \
             _mock_rss(500_000_000, 600_000_000, 645_000_000):
            with pytest.raises(MemoryError, match="Estimated memory"):
                check_memory(model, bench)

    def test_warns_at_high_utilization(self):
        # PLS: baseline=200MB, after probe=800MB, after 2nd=810MB
        # forward=600MB, marginal=10MB/stim, n_stimuli=10
        # total = 200MB + (600MB + 100MB) * 1.5 = 1.25GB. system=1.4GB. 89%. Warn.
        model = FakeModel(activation_nbytes=1000)
        bench = FakeBenchmark(identifier='test-pls', n_stimuli=10)
        with patch('brainscore_core.memory.get_available_memory', return_value=1_200_000_000), \
             _mock_rss(200_000_000, 800_000_000, 890_000_000):
            with pytest.warns(ResourceWarning, match="OOM risk"):
                check_memory(model, bench)

    def test_graceful_when_probe_fails(self):
        model = FailingModel()
        bench = FakeBenchmark(n_stimuli=10)
        with patch('brainscore_core.memory.get_available_memory', return_value=16_000_000_000), \
             _mock_rss(100_000_000):
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

    def test_custom_safety_factor(self):
        # baseline=100MB, forward=100MB, marginal=1MB/stim, n_stimuli=10
        # safety=1.0: total = 100 + (100 + 10) * 1.0 = 210MB. Available+baseline=250MB. Pass.
        # safety=2.0: total = 100 + (100 + 10) * 2.0 = 320MB. Available+baseline=250MB. Fail.
        model = FakeModel(activation_nbytes=1000)
        bench = FakeBenchmark(n_stimuli=10)
        with patch('brainscore_core.memory.get_available_memory', return_value=150_000_000), \
             _mock_rss(100_000_000, 200_000_000, 209_000_000):
            check_memory(model, bench, safety_factor=1.0)  # passes

        with patch('brainscore_core.memory.get_available_memory', return_value=150_000_000), \
             _mock_rss(100_000_000, 200_000_000, 209_000_000):
            with pytest.raises(MemoryError):
                check_memory(model, bench, safety_factor=2.0)

    def test_error_message_includes_identifiers(self):
        model = FakeModel(identifier='big-vit', activation_nbytes=100_000)
        bench = FakeBenchmark(identifier='MajajHong2015-pls', n_stimuli=1000)
        with patch('brainscore_core.memory.get_available_memory', return_value=1_500_000_000), \
             _mock_rss(500_000_000, 600_000_000, 645_000_000):
            with pytest.raises(MemoryError, match="big-vit") as exc_info:
                check_memory(model, bench)
            assert "MajajHong2015" in str(exc_info.value)

    def test_marginal_scales_with_stimuli(self):
        # baseline=100MB, forward=50MB, marginal=5MB/stim
        # 10 stimuli: total = 100 + (50 + 50) * 1.5 = 250MB. system=10GB. Pass.
        model = FakeModel(activation_nbytes=10_000)
        bench_small = FakeBenchmark(identifier='test-pls', n_stimuli=10)
        with patch('brainscore_core.memory.get_available_memory', return_value=10_000_000_000), \
             _mock_rss(100_000_000, 150_000_000, 195_000_000):
            check_memory(model, bench_small)  # passes

        # 5000 stimuli: total = 100 + (50 + 25000) * 1.5 ≈ 37GB. system=500MB. Fail.
        bench_large = FakeBenchmark(identifier='test-pls', n_stimuli=5000)
        with patch('brainscore_core.memory.get_available_memory', return_value=400_000_000), \
             _mock_rss(100_000_000, 150_000_000, 195_000_000):
            with pytest.raises(MemoryError):
                check_memory(model, bench_large)

    def test_auto_safety_factor_by_category(self):
        """Default safety_factor is chosen by metric category."""
        model = FakeModel(activation_nbytes=1000)
        bench_pls = FakeBenchmark(identifier='test-pls', n_stimuli=10)
        bench_cv = FakeBenchmark(identifier='test-ridgecv', n_stimuli=10)
        # baseline=100MB, forward=400MB, marginal=1MB
        # PLS: 100 + (400+10)*1.5 = 715MB < 1GB system. Pass.
        # RidgeCV: 100 + (400+10+metric)*3.0 > 1GB system. Fail.
        with patch('brainscore_core.memory.get_available_memory', return_value=900_000_000), \
             _mock_rss(100_000_000, 500_000_000, 509_000_000):
            check_memory(model, bench_pls)  # passes at 1.5x

        with patch('brainscore_core.memory.get_available_memory', return_value=900_000_000), \
             _mock_rss(100_000_000, 500_000_000, 509_000_000):
            with pytest.raises(MemoryError):
                check_memory(model, bench_cv)  # fails at 3.0x
