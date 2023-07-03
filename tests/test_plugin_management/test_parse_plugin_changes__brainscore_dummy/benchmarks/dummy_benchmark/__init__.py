from brainscore_dummy import benchmark_registry

from .benchmark import DummyBenchmark  # doesn't exist, but we only parse and never run this file and never run it

benchmark_registry['dummy-benchmark'] = DummyBenchmark