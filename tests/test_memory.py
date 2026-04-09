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
)
from brainscore_core.model_interface import UnifiedModel


# ── Helpers ──────────────────────────────────────────────────────────

class FakeModel(UnifiedModel):

    def __init__(self, identifier='test-model', modalities=None,
                 region_layer_map=None, process_result=None):
        self._id = identifier
        self._modalities = modalities or {'vision'}
        self._rlm = region_layer_map or {}
        self._process_result = process_result

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


class FakeBenchmark:

    def __init__(self, identifier='test-bench', n_stimuli=100,
                 batch_size=None, n_targets=None, expected_feature_dim=None):
        self.identifier = identifier
        self.stimulus_set = FakeStimulusSet(n=n_stimuli)
        if batch_size is not None:
            self.batch_size = batch_size
        if n_targets is not None:
            self.n_targets = n_targets
        if expected_feature_dim is not None:
            self.expected_feature_dim = expected_feature_dim


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


# ── estimate_metric_memory ───────────────────────────────────────────

class TestEstimateMetricMemory:

    def test_zero_stimuli(self):
        bench = FakeBenchmark(n_stimuli=0)
        assert estimate_metric_memory(bench) == 0

    def test_uses_defaults(self):
        bench = FakeBenchmark(n_stimuli=100)
        # default n_targets=100, n_features=1000
        # design = 100 * 1000 * 4 = 400_000
        # target = 100 * 100 * 4 = 40_000
        # intermediate = 400_000 * 3 = 1_200_000
        # total = 400_000 + 40_000 + 1_200_000 = 1_640_000
        assert estimate_metric_memory(bench) == 1_640_000

    def test_custom_targets_and_features(self):
        bench = FakeBenchmark(n_stimuli=50, n_targets=20000,
                              expected_feature_dim=2048)
        # design = 50 * 2048 * 4 = 409_600
        # target = 50 * 20000 * 4 = 4_000_000
        # intermediate = 409_600 * 3 = 1_228_800
        # total = 409_600 + 4_000_000 + 1_228_800 = 5_638_400
        assert estimate_metric_memory(bench) == 5_638_400

    def test_no_stimulus_set(self):
        bench = MagicMock(spec=[])  # no attributes at all
        assert estimate_metric_memory(bench) == 0


# ── check_memory ─────────────────────────────────────────────────────

class TestCheckMemory:

    def test_passes_when_plenty_of_memory(self):
        model = FakeModel()
        bench = FakeBenchmark(n_stimuli=10)
        # Mock: available=16GB, peak barely moves
        with patch('brainscore_core.memory.get_available_memory', return_value=16_000_000_000), \
             patch('brainscore_core.memory.get_peak_memory', side_effect=[100, 200]), \
             patch('brainscore_core.memory.reset_peak_memory'):
            check_memory(model, bench)  # should not raise

    def test_raises_when_insufficient_memory(self):
        model = FakeModel()
        bench = FakeBenchmark(n_stimuli=1000, batch_size=64)
        # Mock: available=1GB, per-stimulus cost=100MB
        with patch('brainscore_core.memory.get_available_memory', return_value=1_000_000_000), \
             patch('brainscore_core.memory.get_peak_memory', side_effect=[0, 100_000_000]), \
             patch('brainscore_core.memory.reset_peak_memory'):
            with pytest.raises(MemoryError, match="Estimated memory"):
                check_memory(model, bench)

    def test_warns_at_high_utilization(self):
        model = FakeModel()
        bench = FakeBenchmark(n_stimuli=10)
        # Mock: available=1GB, estimated will be ~85% of available
        # per_stimulus=100MB, batch_size=1, metric~0, safety=1.5
        # total = 100MB * 1 * 1.5 = 150MB... not enough for 80%
        # Need total > 800MB: per_stimulus=600MB, safety=1.5 => 900MB
        with patch('brainscore_core.memory.get_available_memory', return_value=1_000_000_000), \
             patch('brainscore_core.memory.get_peak_memory', side_effect=[0, 600_000_000]), \
             patch('brainscore_core.memory.reset_peak_memory'):
            with pytest.warns(ResourceWarning, match="OOM risk"):
                check_memory(model, bench)

    def test_graceful_when_probe_fails(self):
        model = FailingModel()
        bench = FakeBenchmark(n_stimuli=10)
        with patch('brainscore_core.memory.get_available_memory', return_value=16_000_000_000), \
             patch('brainscore_core.memory.reset_peak_memory'):
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
        model = FakeModel()
        bench = FakeBenchmark(n_stimuli=10)
        # With safety_factor=1.0, total = 100MB. Available = 150MB. Should pass.
        # With safety_factor=2.0, total = 200MB. Available = 150MB. Should fail.
        with patch('brainscore_core.memory.get_available_memory', return_value=150_000_000), \
             patch('brainscore_core.memory.get_peak_memory', side_effect=[0, 100_000_000]), \
             patch('brainscore_core.memory.reset_peak_memory'):
            check_memory(model, bench, safety_factor=1.0)  # passes

        with patch('brainscore_core.memory.get_available_memory', return_value=150_000_000), \
             patch('brainscore_core.memory.get_peak_memory', side_effect=[0, 100_000_000]), \
             patch('brainscore_core.memory.reset_peak_memory'):
            with pytest.raises(MemoryError):
                check_memory(model, bench, safety_factor=2.0)

    def test_error_message_includes_identifiers(self):
        model = FakeModel(identifier='big-vit')
        bench = FakeBenchmark(identifier='MajajHong2015', n_stimuli=1000,
                              batch_size=64)
        with patch('brainscore_core.memory.get_available_memory', return_value=1_000_000_000), \
             patch('brainscore_core.memory.get_peak_memory', side_effect=[0, 100_000_000]), \
             patch('brainscore_core.memory.reset_peak_memory'):
            with pytest.raises(MemoryError, match="big-vit") as exc_info:
                check_memory(model, bench)
            assert "MajajHong2015" in str(exc_info.value)

    def test_batch_size_affects_estimate(self):
        model = FakeModel()
        # batch_size=1: total = 50MB * 1 * 1.5 = 75MB. Available=200MB. Pass.
        bench_small = FakeBenchmark(n_stimuli=1000, batch_size=1)
        with patch('brainscore_core.memory.get_available_memory', return_value=200_000_000), \
             patch('brainscore_core.memory.get_peak_memory', side_effect=[0, 50_000_000]), \
             patch('brainscore_core.memory.reset_peak_memory'):
            check_memory(model, bench_small)  # passes

        # batch_size=64: total = 50MB * 64 * 1.5 = 4.8GB. Available=200MB. Fail.
        bench_large = FakeBenchmark(n_stimuli=1000, batch_size=64)
        with patch('brainscore_core.memory.get_available_memory', return_value=200_000_000), \
             patch('brainscore_core.memory.get_peak_memory', side_effect=[0, 50_000_000]), \
             patch('brainscore_core.memory.reset_peak_memory'):
            with pytest.raises(MemoryError):
                check_memory(model, bench_large)
