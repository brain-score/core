from brainscore_dummy import benchmark_registry

from .benchmark import DummyBenchmark

benchmark_registry['dummy-benchmark'] = DummyBenchmark