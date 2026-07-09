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
