"""Framework-independent serialization contract; no weight or cache downloads."""
import functools
import os
import subprocess
import sys

import numpy as np
import pytest

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


@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
def test_model_weights_and_buffers_change_fingerprint(dtype):
    import torch
    from brainscore_core.extraction_cache import model_config
    model = torch.nn.Linear(2, 2).to(getattr(torch, dtype))
    model.register_buffer('scale', torch.ones(1, dtype=getattr(torch, dtype)))
    before = fingerprint(model_config(model))
    with torch.no_grad():
        model.weight.mul_(2)
    after = fingerprint(model_config(model))
    assert before != after
    model.scale.add_(1)
    assert after != fingerprint(model_config(model))


def test_unreadable_file_bypasses_cache(tmp_path, monkeypatch):
    from brainscore_core.extraction_cache import file_inputs
    monkeypatch.setenv('RESULTCACHING_DISABLE', '0')
    assert extraction_fingerprint(file_inputs([tmp_path / 'missing']),
                                  cache_identifier='test-cache') is None


def test_disabled_cache_does_not_read_file(tmp_path, monkeypatch):
    from brainscore_core.extraction_cache import file_inputs
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    monkeypatch.setattr('builtins.open', lambda *a, **kw: pytest.fail('read with cache disabled'))
    assert extraction_fingerprint(file_inputs([tmp_path / 'missing']),
                                  cache_identifier='test-cache') is None


def test_unmaterialized_weights_bypass_cache(monkeypatch):
    import torch
    from brainscore_core.extraction_cache import model_config
    monkeypatch.setenv('RESULTCACHING_DISABLE', '0')
    model = torch.nn.Linear(2, 2, device='meta')
    assert extraction_fingerprint(model_config(model)) is None


def test_scoped_hash_reuse_and_tracked_mutations(monkeypatch):
    import torch
    from brainscore_core import extraction_cache as cache
    model = torch.nn.Linear(2, 2, bias=False)
    original = cache._hash_tensor
    reads = []
    def counted(tensor):
        reads.append(tensor.numel())
        return original(tensor)
    monkeypatch.setattr(cache, '_hash_tensor', counted)
    with cache.weight_fingerprint_scope():
        before = fingerprint(cache.model_config(model))
        assert fingerprint(cache.model_config(model)) == before
        assert len(reads) == 1
        with torch.no_grad():
            model.weight.mul_(2)
        after = fingerprint(cache.model_config(model))
        assert after != before
        assert len(reads) == 2
        model.weight.data = model.weight.data * 2
        assert fingerprint(cache.model_config(model)) != after
        assert len(reads) == 3
    # A new run must read weights again even if their tracked state is unchanged.
    with cache.weight_fingerprint_scope():
        fingerprint(cache.model_config(model))
    assert len(reads) == 4


@pytest.mark.parametrize('edit', ['data_alias', 'numpy_alias'])
def test_untracked_edits_between_runs_are_detected(edit):
    import torch
    from brainscore_core import extraction_cache as cache
    model = torch.nn.Linear(2, 2, bias=False)
    with cache.weight_fingerprint_scope():
        before = fingerprint(cache.model_config(model))
    if edit == 'data_alias':
        model.weight.data.mul_(2)
    else:
        model.weight.detach().numpy()[:] *= 2
    with cache.weight_fingerprint_scope():
        assert fingerprint(cache.model_config(model)) != before


def test_scope_restores_state_on_failure_and_does_not_retain_tensors():
    import gc
    import weakref
    import torch
    from brainscore_core import extraction_cache as cache
    tensor = torch.ones(3)
    reference = weakref.ref(tensor)
    with pytest.raises(RuntimeError):
        with cache.weight_fingerprint_scope():
            fingerprint(tensor)
            del tensor
            gc.collect()
            assert reference() is None
            raise RuntimeError('failed run')
    assert cache._weight_hashes.get() is None


def test_inference_tensors_are_not_memoized():
    import torch
    from brainscore_core import extraction_cache as cache
    with torch.inference_mode(), cache.weight_fingerprint_scope():
        tensor = torch.ones(3)
        before = fingerprint(tensor)
        tensor.mul_(2)
        assert fingerprint(tensor) != before
        assert cache._weight_hashes.get() == {}


@pytest.mark.parametrize('shape', [(5, 7), (3, 5, 7)])
def test_noncontiguous_hash_has_bounded_copies_and_matches_dense(shape, monkeypatch):
    import torch
    from brainscore_core import extraction_cache as cache
    value = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape).transpose(0, -1)
    monkeypatch.setattr(cache, '_HASH_CHUNK_BYTES', 16)
    chunks = list(cache._tensor_chunks(value, 4))
    assert all(part.numel() <= 4 for part in chunks)
    joined = torch.cat([part.contiguous().reshape(-1) for part in chunks])
    torch.testing.assert_close(joined, value.contiguous().reshape(-1))
    assert fingerprint(value) == fingerprint(value.contiguous())


def test_direct_extraction_keeps_strict_content_checks():
    import torch
    tensor = torch.ones(2)
    before = fingerprint(tensor)
    tensor.data.mul_(2)
    assert fingerprint(tensor) != before
