"""A2: activation extraction must bypass result_caching while a perturbation is
active, so a lesioned run never returns cached UNperturbed activations."""
import os

from brainscore_core.dispatch import InputDispatcher


class _FakeActivationsModel:
    """Records whether the result cache was disabled during the call."""
    def __init__(self):
        self.seen_disable = 'unset'

    def __call__(self, stimuli, layers):
        self.seen_disable = os.environ.get('RESULTCACHING_DISABLE')
        return 'features'


class _FakeOwner:
    def __init__(self, perturbed):
        self._activations_model = _FakeActivationsModel()
        self._active_perturbations = {'h1': (lambda: None)} if perturbed else {}
        self._preprocessors = {}
        self._model = None
        self._recording_layer = None


def _clean_env():
    os.environ.pop('RESULTCACHING_DISABLE', None)


def test_cache_bypassed_while_perturbed():
    _clean_env()
    owner = _FakeOwner(perturbed=True)
    out = InputDispatcher(owner).extract_for_modality('stim', 'vision', ['layer'])
    assert out == 'features'
    # cache was disabled DURING the perturbed extraction
    assert owner._activations_model.seen_disable == '1'
    # and restored (removed) afterwards
    assert os.environ.get('RESULTCACHING_DISABLE') is None


def test_cache_untouched_when_not_perturbed():
    _clean_env()
    owner = _FakeOwner(perturbed=False)
    InputDispatcher(owner).extract_for_modality('stim', 'vision', ['layer'])
    # normal path: cache flag never set
    assert owner._activations_model.seen_disable is None
    assert os.environ.get('RESULTCACHING_DISABLE') is None


def test_prior_disable_flag_is_restored():
    os.environ['RESULTCACHING_DISABLE'] = 'some.module'
    try:
        owner = _FakeOwner(perturbed=True)
        InputDispatcher(owner).extract_for_modality('stim', 'vision', ['layer'])
        assert owner._activations_model.seen_disable == '1'
        assert os.environ.get('RESULTCACHING_DISABLE') == 'some.module'  # restored
    finally:
        _clean_env()


def test_reentrant_disable_restores_only_at_outermost():
    """Interleaved enter/exit must keep the flag set until the OUTERMOST exit."""
    from brainscore_core.dispatch import _activation_cache_disabled
    _clean_env()
    a, b = _activation_cache_disabled(), _activation_cache_disabled()
    a.__enter__()
    assert os.environ.get('RESULTCACHING_DISABLE') == '1'
    b.__enter__()
    a.__exit__(None, None, None)                       # inner exits first
    assert os.environ.get('RESULTCACHING_DISABLE') == '1'   # still disabled
    b.__exit__(None, None, None)                       # outermost exits
    assert os.environ.get('RESULTCACHING_DISABLE') is None  # restored


def test_enter_blocks_while_the_transition_lock_is_held():
    """The depth/flag transition must run under ``_cache_disable_lock`` so a
    second thread cannot interleave the read-modify of the shared counter.

    Racing two threads to hit the exact bad schedule is GIL-nondeterministic, so
    we prove the invariant directly and deterministically: hold the lock, then a
    worker entering ``_activation_cache_disabled()`` must BLOCK until we release.
    Against a lockless implementation the worker enters immediately and the first
    assert fails — which is exactly the regression this guards."""
    import threading
    import brainscore_core.dispatch as dispatch
    from brainscore_core.dispatch import _activation_cache_disabled
    _clean_env()
    dispatch._cache_disable_depth = 0
    dispatch._cache_disable_prev = None

    entered = threading.Event()

    def worker():
        with _activation_cache_disabled():
            entered.set()

    dispatch._cache_disable_lock.acquire()
    t = threading.Thread(target=worker)
    try:
        t.start()
        # lock held -> the guarded transition cannot start (lockless impl: it can)
        assert not entered.wait(timeout=0.5)
    finally:
        dispatch._cache_disable_lock.release()
    # released -> the worker proceeds and completes cleanly
    assert entered.wait(timeout=2)
    t.join(timeout=2)
    assert dispatch._cache_disable_depth == 0
    assert os.environ.get('RESULTCACHING_DISABLE') is None
