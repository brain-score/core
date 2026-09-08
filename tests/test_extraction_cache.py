"""Framework-independent serialization contract; no weight or cache downloads."""
import functools
import os
import subprocess
import sys

import numpy as np

from brainscore_core.extraction_cache import fingerprint, extraction_fingerprint


def _scale(value, factor=1):
    return value * factor


def test_reconstruction_order_and_partial_parameters():
    a = {'transform': functools.partial(_scale, factor=2), 'sizes': np.array([2, 3])}
    b = {'sizes': np.array([2, 3]), 'transform': functools.partial(_scale, factor=2)}
    assert fingerprint(a) == fingerprint(b)
    b['transform'] = functools.partial(_scale, factor=3)
    assert fingerprint(a) != fingerprint(b)
    assert fingerprint([1, 2]) != fingerprint((1, 2))


def test_key_is_stable_across_processes_and_hash_seeds():
    code = "from brainscore_core.extraction_cache import fingerprint; print(fingerprint({'b': {3, 1}, 'a': [2, 3]}))"
    results = [subprocess.check_output([sys.executable, '-c', code], env={**os.environ, 'PYTHONHASHSEED': seed})
               for seed in ('1', '42')]
    assert results[0] == results[1]


def test_cycles_fail_closed():
    state = {}
    state['self'] = state
    assert extraction_fingerprint(state) is None


def test_stateful_provider_can_exclude_runtime_counters():
    class Provider:
        def __init__(self):
            self.factor = 2
            self.calls = 0
        def cache_config(self):
            return {'factor': self.factor}
    provider = Provider()
    before = fingerprint(provider)
    provider.calls += 1
    assert fingerprint(provider) == before
    provider.factor = 3
    assert fingerprint(provider) != before
