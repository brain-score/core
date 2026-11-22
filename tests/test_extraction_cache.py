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


def test_a_scope_inside_an_active_scope_reuses_its_hashes(monkeypatch):
    import torch
    from brainscore_core import extraction_cache as cache
    model = torch.nn.Linear(2, 2, bias=False)
    original = cache._hash_tensor
    reads = []
    monkeypatch.setattr(cache, '_hash_tensor', lambda tensor: reads.append(1) or original(tensor))
    with cache.weight_fingerprint_scope():
        outer = fingerprint(cache.model_config(model))
        with cache.weight_fingerprint_scope():
            assert fingerprint(cache.model_config(model)) == outer
        assert cache._weight_hashes.get() is not None  # the inner scope left the run's memo open
        assert fingerprint(cache.model_config(model)) == outer
    assert len(reads) == 1
    assert cache._weight_hashes.get() is None


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


@pytest.fixture
def cache_switch_environment():
    """Keep the cache switch tests from changing the environment of later tests."""
    previous = os.environ.get('RESULTCACHING_DISABLE')
    yield
    if previous is None:
        os.environ.pop('RESULTCACHING_DISABLE', None)
    else:
        os.environ['RESULTCACHING_DISABLE'] = previous


def test_reentrant_disable_restores_only_at_outermost(cache_switch_environment):
    """Interleaved enter/exit must keep the flag set until the OUTERMOST exit."""
    from brainscore_core.extraction_cache import activation_cache_disabled
    os.environ.pop('RESULTCACHING_DISABLE', None)
    a, b = activation_cache_disabled(), activation_cache_disabled()
    a.__enter__()
    assert os.environ.get('RESULTCACHING_DISABLE') == '1'
    b.__enter__()
    a.__exit__(None, None, None)                       # inner exits first
    assert os.environ.get('RESULTCACHING_DISABLE') == '1'   # still disabled
    b.__exit__(None, None, None)                       # outermost exits
    assert os.environ.get('RESULTCACHING_DISABLE') is None  # restored


def test_enter_blocks_while_the_transition_lock_is_held(cache_switch_environment):
    """The depth/flag transition must run under ``_cache_disable_lock`` so a
    second thread cannot interleave the read-modify of the shared counter.

    Racing two threads to hit the exact bad schedule is GIL-nondeterministic, so
    we prove the invariant directly and deterministically: hold the lock, then a
    worker entering ``activation_cache_disabled()`` must BLOCK until we release.
    Against a lockless implementation the worker enters immediately and the first
    assert fails — which is exactly the regression this guards."""
    import threading
    import brainscore_core.extraction_cache as cache
    from brainscore_core.extraction_cache import activation_cache_disabled
    os.environ.pop('RESULTCACHING_DISABLE', None)
    cache._cache_disable_depth = 0
    cache._cache_disable_prev = None

    entered = threading.Event()

    def worker():
        with activation_cache_disabled():
            entered.set()

    cache._cache_disable_lock.acquire()
    t = threading.Thread(target=worker)
    try:
        t.start()
        # lock held -> the guarded transition cannot start (lockless impl: it can)
        assert not entered.wait(timeout=0.5)
    finally:
        cache._cache_disable_lock.release()
    # released -> the worker proceeds and completes cleanly
    assert entered.wait(timeout=2)
    t.join(timeout=2)
    assert cache._cache_disable_depth == 0
    assert os.environ.get('RESULTCACHING_DISABLE') is None


@pytest.mark.parametrize('inference', [False, True])
def test_a_part_fingerprint_comes_from_the_same_traversal(monkeypatch, inference):
    """The whole key and one part's key from one pass: no tensor is hashed twice, with or without
    version counters (a tensor made under inference mode has none)."""
    import contextlib
    import torch
    from brainscore_core import extraction_cache as cache
    calls, original = [], cache._hash_tensor
    monkeypatch.setattr(cache, '_hash_tensor', lambda tensor: calls.append(1) or original(tensor))
    with torch.inference_mode() if inference else contextlib.nullcontext():
        weights = [torch.ones(3), torch.arange(4.0)]
    configuration = {'configuration': {'weights': weights, 'batch_size': 2}, 'inputs': ['a', 'b']}
    whole, part = extraction_fingerprint(configuration, part='configuration')
    assert len(calls) == 2
    assert whole == fingerprint(configuration)
    assert part == fingerprint(configuration['configuration'])


def test_a_part_of_a_disabled_or_opaque_configuration_is_none(monkeypatch):
    monkeypatch.setenv('RESULTCACHING_DISABLE', '1')
    assert extraction_fingerprint({'configuration': 1}, cache_identifier='x', part='configuration') == (None, None)
    monkeypatch.delenv('RESULTCACHING_DISABLE')
    assert extraction_fingerprint({'configuration': object()}, part='configuration') == (None, None)


def test_a_store_entry_with_long_names_fits_one_path_component(tmp_path, monkeypatch):
    """A long model identifier and a long table name still give an entry that the file system
    takes: the entry is written once, read back at the second call, and each name under the store
    (with the '.filepart' that the store adds while it saves) fits one path component."""
    import xarray as xr
    from brainscore_core.extraction_cache import store_xarray
    monkeypatch.setenv('RESULTCACHING_HOME', str(tmp_path))
    monkeypatch.delenv('RESULTCACHING_DISABLE', raising=False)
    calls = []

    class Extractor:
        @store_xarray(identifier_ignore=['layers'], combine_fields={'layers': 'layer'})
        def activations(self, identifier, layers, stimuli_identifier, extraction_fingerprint):
            calls.append(list(layers))
            return xr.DataArray(np.zeros((1, len(layers))), dims=['presentation', 'neuroid'],
                                coords={'layer': ('neuroid', list(layers))})

    arguments = dict(identifier='m' * 120, layers=['a'], stimuli_identifier='t' * 150,
                     extraction_fingerprint='v2-' + 'f' * 64)
    Extractor().activations(**arguments)
    Extractor().activations(**arguments)
    assert len(calls) == 1
    assert all(len((path.name + '.filepart').encode()) <= 255 for path in tmp_path.rglob('*'))


def test_a_disable_prefix_reaches_the_arguments_of_a_long_name(tmp_path, monkeypatch):
    """``RESULTCACHING_DISABLE=<prefix>`` is read on the whole name, also when the name is too long
    for one path component: a prefix into the arguments disables the entry (no file, no read)."""
    import xarray as xr
    from brainscore_core.extraction_cache import store_xarray
    monkeypatch.setenv('RESULTCACHING_HOME', str(tmp_path))
    calls = []

    class Extractor:
        @store_xarray(identifier_ignore=['layers'], combine_fields={'layers': 'layer'})
        def activations(self, identifier, layers, stimuli_identifier, extraction_fingerprint):
            calls.append(list(layers))
            return xr.DataArray(np.zeros((1, len(layers))), dims=['presentation', 'neuroid'],
                                coords={'layer': ('neuroid', list(layers))})

    arguments = dict(identifier='m' * 120, layers=['a'], stimuli_identifier='t' * 150,
                     extraction_fingerprint='v2-' + 'f' * 64)
    name = f'{__name__}.Extractor.activations/identifier={"m" * 120}'
    monkeypatch.setenv('RESULTCACHING_DISABLE', name)
    Extractor().activations(**arguments)
    Extractor().activations(**arguments)
    assert len(calls) == 2
    assert not any(path.is_file() for path in tmp_path.rglob('*'))
