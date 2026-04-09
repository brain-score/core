import pytest
from unittest.mock import MagicMock, call, patch

from brainscore_core.benchmarks import Benchmark, BenchmarkBase, score_benchmark
from brainscore_core.metrics import Score


# ---------------------------------------------------------------------------
# Minimal concrete Benchmark implementations for testing
# ---------------------------------------------------------------------------

class _DummyBenchmark(BenchmarkBase):
    """Minimal benchmark that returns a fixed score and does NOT override preallocate_memory."""

    def __init__(self, score_value=0.5):
        ceiling = Score(1.0)
        super().__init__(identifier='dummy', ceiling=ceiling, version=1, parent='neural')
        self._score_value = score_value
        self.called_with = []

    def __call__(self, candidate):
        self.called_with.append(candidate)
        return Score(self._score_value)


class _MemoryCheckingBenchmark(_DummyBenchmark):
    """Benchmark that overrides preallocate_memory — tracks calls and optionally raises."""

    def __init__(self, score_value=0.5, raise_oom=False):
        super().__init__(score_value=score_value)
        self.preallocate_calls = []
        self._raise_oom = raise_oom

    def preallocate_memory(self, candidate):
        self.preallocate_calls.append(candidate)
        if self._raise_oom:
            raise MemoryError("Estimated 99 GB needed, 8 GB available.")


# ---------------------------------------------------------------------------
# Tests: Benchmark.preallocate_memory default
# ---------------------------------------------------------------------------

class TestPreallocateMemoryDefault:
    def test_noop_returns_none(self):
        """Default preallocate_memory is a no-op and returns None."""
        benchmark = _DummyBenchmark()
        candidate = MagicMock()
        result = benchmark.preallocate_memory(candidate)
        assert result is None

    def test_noop_does_not_call_candidate(self):
        """Default preallocate_memory does not interact with the candidate at all."""
        benchmark = _DummyBenchmark()
        candidate = MagicMock()
        benchmark.preallocate_memory(candidate)
        candidate.assert_not_called()

    def test_noop_on_any_candidate(self):
        """Default preallocate_memory accepts any candidate without error."""
        benchmark = _DummyBenchmark()
        for candidate in [None, 42, "string", object()]:
            benchmark.preallocate_memory(candidate)  # should not raise


# ---------------------------------------------------------------------------
# Tests: score_benchmark — ordering and delegation
# ---------------------------------------------------------------------------

class TestScoreBenchmark:
    def test_returns_score_from_benchmark(self):
        benchmark = _DummyBenchmark(score_value=0.8)
        candidate = MagicMock()
        result = score_benchmark(benchmark, candidate)
        assert float(result) == pytest.approx(0.8)

    def test_calls_benchmark_with_candidate(self):
        benchmark = _DummyBenchmark()
        candidate = MagicMock()
        score_benchmark(benchmark, candidate)
        assert benchmark.called_with == [candidate]

    def test_preallocate_called_before_benchmark(self):
        """preallocate_memory must be called before __call__."""
        call_order = []

        class _OrderTracking(_DummyBenchmark):
            def preallocate_memory(self, candidate):
                call_order.append('preallocate')

            def __call__(self, candidate):
                call_order.append('score')
                return Score(0.5)

        benchmark = _OrderTracking()
        score_benchmark(benchmark, MagicMock())
        assert call_order == ['preallocate', 'score']

    def test_preallocate_receives_candidate(self):
        benchmark = _MemoryCheckingBenchmark()
        candidate = MagicMock()
        score_benchmark(benchmark, candidate)
        assert benchmark.preallocate_calls == [candidate]

    def test_raises_memory_error_before_scoring(self):
        """If preallocate_memory raises MemoryError, __call__ must never execute."""
        benchmark = _MemoryCheckingBenchmark(raise_oom=True)
        candidate = MagicMock()
        with pytest.raises(MemoryError):
            score_benchmark(benchmark, candidate)
        assert benchmark.called_with == [], "benchmark.__call__ should not have been invoked"

    def test_memory_error_message_propagates(self):
        benchmark = _MemoryCheckingBenchmark(raise_oom=True)
        with pytest.raises(MemoryError, match="99 GB needed"):
            score_benchmark(benchmark, MagicMock())

    def test_no_override_still_scores(self):
        """Benchmark without preallocate_memory override runs normally."""
        benchmark = _DummyBenchmark(score_value=0.42)
        result = score_benchmark(benchmark, MagicMock())
        assert float(result) == pytest.approx(0.42)

    def test_preallocate_called_exactly_once(self):
        benchmark = _MemoryCheckingBenchmark()
        score_benchmark(benchmark, MagicMock())
        assert len(benchmark.preallocate_calls) == 1

    def test_multiple_candidates_independent(self):
        """Each score_benchmark call is independent."""
        benchmark = _MemoryCheckingBenchmark(score_value=0.7)
        c1, c2 = MagicMock(), MagicMock()
        score_benchmark(benchmark, c1)
        score_benchmark(benchmark, c2)
        assert benchmark.preallocate_calls == [c1, c2]
        assert benchmark.called_with == [c1, c2]


# ---------------------------------------------------------------------------
# Tests: skip env var (validates the env var is respected in memory.py,
# tested here at the interface level via a mock)
# ---------------------------------------------------------------------------

class TestSkipEnvVar:
    def test_skip_flag_bypasses_oom(self, monkeypatch):
        """BRAINSCORE_SKIP_MEMORY_CHECK=1 should allow scoring even when preallocate raises."""
        monkeypatch.setenv('BRAINSCORE_SKIP_MEMORY_CHECK', '1')

        class _EnvAwareBenchmark(_DummyBenchmark):
            def preallocate_memory(self, candidate):
                import os
                if os.environ.get('BRAINSCORE_SKIP_MEMORY_CHECK', '0') == '1':
                    return
                raise MemoryError("would OOM")

        benchmark = _EnvAwareBenchmark(score_value=0.6)
        result = score_benchmark(benchmark, MagicMock())
        assert float(result) == pytest.approx(0.6)
