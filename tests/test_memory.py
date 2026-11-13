import pytest
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
from brainscore_core.contract import UnifiedModel
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


# ── Helpers ──────────────────────────────────────────────────────────

def FakeResult(nbytes=8000, n_features=1000):
    """A one-presentation NeuroidAssembly of the given width (nbytes is unused)."""
    import numpy as np
    del nbytes
    return NeuroidAssembly(np.zeros((1, n_features), dtype='float32'),
                           dims=['presentation', 'neuroid'],
                           coords={'stimulus_id': ('presentation', [0]),
                                   'neuroid_id': ('neuroid', list(range(n_features)))})


class FakeModel(UnifiedModel):

    def __init__(self, identifier='test-model', in_channels=None,
                 region_layer_map=None, process_result=None,
                 activation_nbytes=8000, n_features=1000):
        self._id = identifier
        self._in_channels = in_channels or {'vision:image'}
        # A model that records somewhere; the probe reads it from the declared output.
        self._rlm = region_layer_map if region_layer_map is not None else {'IT': 'layer'}
        self._process_result = (process_result if process_result is not None
                                else FakeResult(nbytes=activation_nbytes, n_features=n_features))

    @property
    def identifier(self) -> str:
        return self._id

    @property
    def region_layer_map(self) -> Dict[str, str]:
        return self._rlm

    @property
    def in_channels(self) -> Set[str]:
        return self._in_channels

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
    """A layer-aware subject that only accepts process() AFTER start_recording
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
        return FakeResult(n_features=self._per * len(self._recorded))


def FakeStimulusSet(n=10):
    """A real n-row stimulus table, since the probe runs through a session."""
    import pandas as pd
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': [f's{i}' for i in range(n)],
                                        'image_path': [f'/tmp/s{i}.png' for i in range(n)]}))
    stimuli.identifier = f'fake-{n}'
    return stimuli


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


# ── what estimate_metric_memory reads from a benchmark ──────────────

class TestMetricInputsFromTheBenchmark:
    """The metric kind, the target count and the stimulus count come from the benchmark.

    Each case compares the estimate with the estimate for an explicit value
    (``category=``, ``n_targets``), so the test goes through the public function.
    """

    def test_each_category_gives_its_own_estimate(self):
        bench = FakeBenchmark(identifier='plain', n_stimuli=100,
                              assembly=FakeAssembly(n_neuroids=600))
        estimates = {category: estimate_metric_memory(bench, category=category)
                     for category in ('pls', 'ridge', 'ridgecv', 'rsa', 'behavioral')}
        assert len(set(estimates.values())) == 5
        assert estimates['behavioral'] == 0

    @pytest.mark.parametrize('identifier, has_assembly, category', [
        ('MajajHong2015.IT-pls', True, 'pls'),
        ('Allen2022_fmri_surface.IT-ridge', True, 'ridge'),
        ('Papale2025.IT-ridgecv', True, 'ridgecv'),
        ('something-ridgecv-split', True, 'ridgecv'),   # ridgecv wins over ridge
        ('Allen2022_fmri_surface.IT-rdm', True, 'rsa'),
        ('Rajalingham2018-i2n', False, 'behavioral'),   # no assembly: behavioral
        ('SomeBench', True, 'pls'),                     # neural default
    ])
    def test_category_comes_from_the_identifier_or_the_assembly(self, identifier, has_assembly,
                                                                 category):
        bench = FakeBenchmark(identifier=identifier, n_stimuli=100,
                              assembly=FakeAssembly(n_neuroids=600) if has_assembly else None)
        assert estimate_metric_memory(bench) == estimate_metric_memory(bench, category=category)

    def test_a_benchmark_without_identifier_is_behavioral(self):
        assert estimate_metric_memory(MagicMock(spec=[])) == 0

    @pytest.mark.parametrize('attribute', ['_assembly', 'train_assembly'])
    def test_target_count_is_read_from_the_assembly(self, attribute):
        bench = FakeBenchmark(identifier='x-pls', n_stimuli=100)
        setattr(bench, attribute, FakeAssembly(n_neuroids=3000))
        declared = FakeBenchmark(identifier='x-pls', n_stimuli=100, n_targets=3000)
        assert estimate_metric_memory(bench) == estimate_metric_memory(declared)

    def test_target_count_defaults_to_100(self):
        default = FakeBenchmark(identifier='x-pls', n_stimuli=100)
        assert estimate_metric_memory(default) == estimate_metric_memory(
            FakeBenchmark(identifier='x-pls', n_stimuli=100, n_targets=100))
        assert estimate_metric_memory(default) != estimate_metric_memory(
            FakeBenchmark(identifier='x-pls', n_stimuli=100, n_targets=3000))

    def test_stimulus_count_comes_from_the_stimulus_set_or_the_assembly(self):
        from types import SimpleNamespace
        from_set = FakeBenchmark(identifier='x-rdm', n_stimuli=300)
        from_assembly = SimpleNamespace(identifier='x-rdm', stimulus_set=None,
                                        _assembly=FakeAssembly(n_stimuli=300))
        assert estimate_metric_memory(from_assembly) == estimate_metric_memory(from_set) > 0


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

    # ── ExecutionPlan (DECLARED-grounded) path ───────────────────────

    def test_execution_plan_skips_the_probe(self):
        # A declared plan with feature_width means no probe: CountingFailingModel
        # would raise if probed, but the plan drives the estimate.
        from brainscore_core.execution_plan import ExecutionPlan
        model = CountingFailingModel()
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=10)
        bench.execution_plan = ExecutionPlan(n_extraction_presentations=1000,
                                             feature_width=100000)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=300_000_000), \
             _mock_memory(500_000_000):
            with pytest.raises(MemoryError, match="DECLARED-grounded"):
                check_memory(model, bench)
        assert model.process_calls == 0

    def test_execution_plan_result_is_declared_not_approximate(self, caplog):
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
        assert any('DECLARED-grounded' in m for m in msgs)
        assert not any('APPROXIMATE' in m for m in msgs)

    def test_feature_dim_fills_the_plan_not_bypasses_it(self):
        # passing feature_dim must FILL the plan's raw width, not disable
        # the plan. A billion-row plan + feature_dim=1 must still raise (plan
        # cardinality drives it); if feature_dim bypassed the plan, it would pass
        # off the 10-row stimulus set.
        from brainscore_core.execution_plan import ExecutionPlan
        model = FakeModel()
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=10)
        bench.execution_plan = ExecutionPlan(n_extraction_presentations=1_000_000_000)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=1_000_000), \
             _mock_memory(200_000_000):
            with pytest.raises(MemoryError, match="n_presentations: 1000000000"):
                check_memory(model, bench, feature_dim=1)

    def test_complete_plan_needs_no_stimulus_set(self):
        # a plan with feature_width declared needs no stimulus set — the
        # estimate must still run (and here, raise), not skip early.
        from brainscore_core.execution_plan import ExecutionPlan
        model = FakeModel()
        bench = MagicMock(spec=['identifier', 'execution_plan'])
        bench.identifier = 'test-ridge'
        bench.execution_plan = ExecutionPlan(n_extraction_presentations=1_000_000,
                                             feature_width=100000)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=1_000_000), \
             _mock_memory(200_000_000):
            with pytest.raises(MemoryError):
                check_memory(model, bench)

    def test_plan_probe_uses_declared_target_and_stimuli(self):
        # the plan names the recording_target and a probe input of the
        # shape the benchmark processes, so an image-only model that rejects the raw
        # (video) stimulus still probes — recording the RIGHT region.
        from brainscore_core.execution_plan import ExecutionPlan
        recorded = {}

        one_frame = FakeStimulusSet(n=1)
        one_frame.identifier = 'one-frame'

        class _FrameOnlyModel(FakeModel):
            def start_recording(self, target, time_bins=None, recording_type=None):
                recorded['target'] = target
            def process(self, stimuli):
                if getattr(stimuli, 'identifier', None) != 'one-frame':
                    raise RuntimeError("image-only model can't process a video row")
                return FakeResult(n_features=4321)

        model = _FrameOnlyModel(region_layer_map={'V1': 'a', 'IT': 'b'})
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=10)  # raw = videos
        bench.execution_plan = ExecutionPlan(
            n_extraction_presentations=100000,
            recording_target='IT', probe_stimuli=one_frame)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=1_000_000_000), \
             _mock_memory(200_000_000):
            with pytest.raises(MemoryError, match=r"n_features: 4321"):
                check_memory(model, bench)
        assert recorded['target'] == 'IT'  # the declared target, not the first region

    def test_plan_metric_category_override(self):
        # the plan overrides identifier-based category detection. Identifier
        # says nothing (-> PLS by default), but declaring ridgecv with many alphas
        # blows the metric up and raises where PLS would pass.
        from brainscore_core.execution_plan import ExecutionPlan
        model = FakeModel()
        bench = FakeBenchmark(identifier='mystery-bench', n_stimuli=10,
                              assembly=FakeAssembly(n_neuroids=5000))
        bench.execution_plan = ExecutionPlan(n_extraction_presentations=1000,
                                             feature_width=1000,
                                             metric_category='ridgecv')
        with patch('brainscore_core.memory.get_host_available_memory', return_value=300_000_000), \
             _mock_memory(200_000_000):
            with pytest.raises(MemoryError, match=r"metric \[ridgecv\]"):
                check_memory(model, bench)

    def test_plan_runs_ceiling_metric_flag_is_load_bearing(self):
        # The ceiling metric (F = n_targets = 50000) has coef ~ 50000^2*8 ~ 20 GB
        # and dominates. With runs_ceiling_metric=True the estimate raises; dropping
        # the ceiling (False) leaves only the tiny model metric, which fits. If the
        # flag were ignored, the False case would still raise.
        from brainscore_core.execution_plan import ExecutionPlan
        model = FakeModel()
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=10,
                              assembly=FakeAssembly(n_neuroids=50000))
        common = dict(n_extraction_presentations=1000, feature_width=100,
                      metric_feature_width=100, metric_observations=1000)
        bench.execution_plan = ExecutionPlan(**common, runs_ceiling_metric=True)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=2_000_000_000), \
             _mock_memory(200_000_000):
            with pytest.raises(MemoryError):
                check_memory(model, bench)
        bench.execution_plan = ExecutionPlan(**common, runs_ceiling_metric=False)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=2_000_000_000), \
             _mock_memory(200_000_000):
            check_memory(model, bench)  # passes only because the ceiling was dropped

    def test_complete_plan_does_not_touch_stimulus_set(self):
        # A plan with feature_width declared must NOT access the (lazy/expensive)
        # stimulus set, and a real MemoryError from that property must not be
        # swallowed if it ever is touched.
        from brainscore_core.execution_plan import ExecutionPlan

        class _Bench:
            identifier = 'test-ridge'
            execution_plan = ExecutionPlan(n_extraction_presentations=100,
                                           feature_width=1000)
            @property
            def stimulus_set(self):
                raise AssertionError("stimulus_set must not be accessed")

        with patch('brainscore_core.memory.get_host_available_memory', return_value=16_000_000_000), \
             _mock_memory(200_000_000):
            check_memory(FakeModel(), _Bench())  # no AssertionError -> never touched

    def test_probed_zero_width_is_not_accepted(self, caplog):
        import logging
        # A degenerate (1, 0) probe result must be treated as unresolved (skip
        # loudly), not a zero-width, zero-memory pass.
        model = FakeModel(process_result=FakeResult(n_features=0))
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=10)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=16_000_000_000), \
             _mock_memory(200_000_000):
            with caplog.at_level(logging.WARNING, logger='brainscore_core.memory'):
                check_memory(model, bench)
        assert any('could NOT run' in r.getMessage() for r in caplog.records)

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

    def test_execution_plan_as_method_is_honored(self):
        from brainscore_core.execution_plan import ExecutionPlan
        model = CountingFailingModel()
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=10)
        bench.execution_plan = lambda: ExecutionPlan(
            n_extraction_presentations=1000, feature_width=100000)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=300_000_000), \
             _mock_memory(500_000_000):
            with pytest.raises(MemoryError, match="DECLARED-grounded"):
                check_memory(model, bench)
        assert model.process_calls == 0

    def test_execution_plan_without_feature_width_probes_for_raw(self):
        # feature_width omitted -> the DECLARED path probes the model for the raw
        # (stimulus-invariant) width, then uses the declared cardinality. The
        # message reflects the PROBED 50000, proving the probe filled it in.
        from brainscore_core.execution_plan import ExecutionPlan
        model = FakeModel(n_features=50000)
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=10)
        bench.execution_plan = ExecutionPlan(n_extraction_presentations=100000)
        with patch('brainscore_core.memory.get_host_available_memory', return_value=1_000_000_000), \
             _mock_memory(200_000_000):
            with pytest.raises(MemoryError, match=r"n_features: 50000 \[ExecutionPlan\]"):
                check_memory(model, bench)

    def test_non_plan_execution_plan_rejected(self):
        model = FakeModel()
        bench = FakeBenchmark(identifier='test-ridge', n_stimuli=10)
        bench.execution_plan = {'not': 'a plan'}
        with pytest.raises(TypeError, match="must be an ExecutionPlan"):
            check_memory(model, bench)

    def test_shapeless_probe_result_warns_and_skips(self, caplog):
        import logging
        # process() returns a response with no presentation axis -> the session
        # cannot carry it as a neural response -> feature width unreadable -> skip loudly.
        import numpy as np
        model = FakeModel(process_result=NeuroidAssembly(
            np.zeros((5, 3)), dims=['time', 'neuroid'],
            coords={'time': ('time', list(range(5))), 'neuroid_id': ('neuroid', [0, 1, 2])}))
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


class TestProbeCacheIsolation:
    """The memory probe must not share an activation-cache key with the real run.

    Slicing a StimulusSet keeps the parent's identifier, and activation caches
    key on that identifier rather than on the rows. Sharing it broke scoring in
    both directions: cold cache, the probe stored a one-row assembly the full run
    then read back; warm cache, the probe read the full assembly. Either way the
    presentation dimension disagreed with the stimulus set and packaging raised.
    """

    @staticmethod
    def _stimulus_set(n=4):
        import pandas as pd
        from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
        s = StimulusSet(pd.DataFrame({'stimulus_id': [f'i{i}' for i in range(n)]}))
        s.identifier = 'hvm-public'
        s.stimulus_paths = {f'i{i}': f'/tmp/i{i}.png' for i in range(n)}
        return s

    @staticmethod
    def _probe_identifiers(stimulus_set, runs=1):
        """Identifiers of the stimuli each probe session shows the subject."""
        import numpy as np
        from types import SimpleNamespace
        from brainscore_core.contract import Subject
        from brainscore_core.streaming import StreamEvent

        class Watching(Subject):
            identifier = 'watching'
            in_channels = {'vision:image'}
            out_channels = {'neural:IT'}
            seen = []

            def participate(self, session):
                self.seen.append(getattr(session.stimuli, 'identifier', None))
                n = len(list(iter(session.next_input, None)))
                session.respond(StreamEvent('neural:IT', NeuroidAssembly(
                    np.zeros((n, 4)), dims=['presentation', 'neuroid'],
                    coords={'stimulus_id': ('presentation', list(range(n))),
                            'neuroid_id': ('neuroid', list(range(4)))}), None))

        subject = Watching()
        bench = SimpleNamespace(identifier='bench', stimulus_set=stimulus_set, region='IT')
        with patch('brainscore_core.memory.get_host_available_memory', return_value=16_000_000_000), \
             _mock_memory(100):
            for _ in range(runs):
                check_memory(subject, bench)
        return subject.seen

    def test_probe_identifier_differs_from_the_parent(self):
        parent = self._stimulus_set()
        seen, = self._probe_identifiers(parent)
        assert seen != parent.identifier
        assert parent.identifier == 'hvm-public'      # parent untouched

    def test_probe_identifier_is_derived_and_stable(self):
        assert self._probe_identifiers(self._stimulus_set(), runs=2) == [
            'hvm-public-memory-probe', 'hvm-public-memory-probe']

    def test_identifierless_stimulus_set_passes_through(self):
        import pandas as pd
        from brainscore_core.memory import _isolate_probe_identifier
        plain = pd.DataFrame({'stimulus_id': ['a', 'b']})
        assert _isolate_probe_identifier(plain.iloc[:1], plain) is not None

    def test_unsliceable_stimulus_set_is_not_renamed(self):
        """When the probe *is* the full set, renaming it would rename the run."""
        from brainscore_core.memory import _isolate_probe_identifier
        parent = self._stimulus_set()
        assert _isolate_probe_identifier(parent, parent).identifier == 'hvm-public'


class TestProbeSafeRegions:
    """``probe_safe_regions`` narrows the probe only where a benchmark names no region.

    A subject that declares it (the vision legacy adapter: its committed
    layers) is probed only at those regions on a benchmark that declares no
    outputs, so the probe never starts a lazy layer search. Every other case
    keeps the existing probe exactly.
    """

    def _subject(self, region_layer_map=None, probe_safe_regions=None, failing=()):
        """A subject declaring neural:V4, neural:EXTRA, neural:IT that counts its sessions."""
        import numpy as np
        from brainscore_core.contract import Subject
        from brainscore_core.streaming import StreamEvent
        from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly

        class Counting(Subject):
            identifier = 'counting'
            in_channels = {'vision'}
            out_channels = {'neural:V4', 'neural:EXTRA', 'neural:IT', 'behavior'}
            required_channels = set()

            def __init__(self):
                self.sessions = []

            def participate(self, session):
                channel, = session.outputs
                self.sessions.append(channel)
                if channel in failing:
                    raise RuntimeError(f'{channel} unavailable')
                n = 0
                while session.next_input() is not None:
                    n += 1
                session.respond(StreamEvent(channel, NeuroidAssembly(
                    np.zeros((n, 9)), dims=['presentation', 'neuroid'],
                    coords={'stimulus_id': ('presentation', list(range(n))),
                            'neuroid_id': ('neuroid', list(range(9)))}), None))

        subject = Counting()
        if region_layer_map is not None:
            subject.region_layer_map = region_layer_map
        if probe_safe_regions is not None:
            subject.probe_safe_regions = probe_safe_regions
        return subject

    def _bench(self, **declarations):
        from types import SimpleNamespace
        return SimpleNamespace(identifier='bench', stimulus_set=FakeStimulusSet(n=3), **declarations)

    def _check(self, subject, benchmark):
        """Run the memory check; return the output channel of each probe session, in order."""
        with patch('brainscore_core.memory.get_host_available_memory', return_value=16_000_000_000), \
             _mock_memory(100):
            check_memory(subject, benchmark)
        return subject.sessions

    def _probe_order(self, region_layer_map=None, probe_safe_regions=None, **declarations):
        """Every region the probe would try, in order: each one fails, so each is tried."""
        subject = self._subject(region_layer_map, probe_safe_regions,
                                failing=('neural:V4', 'neural:EXTRA', 'neural:IT'))
        return self._check(subject, self._bench(**declarations))

    def test_all_committed_regions_safe_probes_as_before(self):
        full = {'V4': 'a', 'EXTRA': 'b', 'IT': 'c'}
        assert (self._probe_order(full, probe_safe_regions=list(full)) == self._probe_order(full)
                == ['neural:V4', 'neural:EXTRA', 'neural:IT'])
        plain = self._subject(full)
        declared = self._subject(full, probe_safe_regions=list(full))
        assert self._check(declared, self._bench()) == self._check(plain, self._bench()) == ['neural:V4']

    def test_no_safe_region_means_no_probe_on_an_undeclared_benchmark(self, caplog):
        import logging
        subject = self._subject({}, probe_safe_regions=[])  # a lazy map before any search
        with caplog.at_level(logging.WARNING, logger='brainscore_core.memory'):
            check_memory(subject, FakeBenchmark(identifier='Kar2019-ost', n_stimuli=4))
        assert subject.sessions == []
        assert any('probe_safe_regions' in r.getMessage() for r in caplog.records)

    def test_only_safe_regions_are_probed_on_an_undeclared_benchmark(self):
        assert self._probe_order({'IT': 'c'}, probe_safe_regions=['IT']) == ['neural:IT']

    def test_subject_without_the_declaration_keeps_the_fallback(self):
        # No map: every declared neural region, sorted.
        assert self._probe_order() == ['neural:EXTRA', 'neural:IT', 'neural:V4']
        # Partial map: map order first, then the remaining declared regions.
        assert self._probe_order({'V4': 'a'}) == ['neural:V4', 'neural:EXTRA', 'neural:IT']
        subject = self._subject({'V4': 'a'}, failing=('neural:V4',))
        assert self._check(subject, self._bench()) == ['neural:V4', 'neural:EXTRA']

    def test_declared_benchmark_ignores_probe_safe_regions(self):
        assert self._probe_order({}, probe_safe_regions=[], region='IT') == ['neural:IT']
        assert self._probe_order({}, probe_safe_regions=[],
                                 requested_output_channels={'neural:IT', 'behavior'}) == ['neural:IT']

    @pytest.mark.parametrize('requested', [frozenset({'behavior'}), frozenset()])
    def test_declared_outputs_without_a_neural_channel_mean_no_probe(self, requested):
        subject = self._subject({'V4': 'a', 'IT': 'c'}, probe_safe_regions=['V4', 'IT'])
        assert self._check(subject, self._bench(requested_output_channels=requested)) == []

    def test_region_keeps_precedence_over_requested_channels(self):
        assert self._probe_order({'V4': 'a', 'IT': 'c'}, region='IT',
                                 requested_output_channels={'neural:V4'}) == ['neural:IT']
        assert self._probe_order({'V4': 'a', 'IT': 'c'}, region='IT',
                                 requested_output_channels={'behavior'}) == ['neural:IT']
