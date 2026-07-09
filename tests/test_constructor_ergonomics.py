"""C4: capability functions are explicit keyword-only params (visible in the
signature/help), and the positional-tail footgun warns."""
import inspect
import warnings

import pytest

from brainscore_core import BrainScoreModel


def _f(*a, **k):
    return None


def test_explicit_capability_kwargs_route_correctly():
    m = BrainScoreModel('id', None, {}, {}, None,
                        action_fn=_f, state_change_fn=_f,
                        generation_fn=_f, behavioral_readout_layer='layer.10')
    assert m._action_fn is _f
    assert m._state_change_fn is _f
    assert m._generation_fn is _f
    assert m._behavioral_readout_layer == 'layer.10'


def test_capability_params_are_in_the_signature():
    params = inspect.signature(BrainScoreModel.__init__).parameters
    for name in ('behavioral_readout_layer', 'generation_fn', 'action_fn', 'state_change_fn'):
        assert name in params, f"{name} should be an explicit keyword param"
        assert params[name].kind == inspect.Parameter.KEYWORD_ONLY


def test_constructor_and_process_are_documented():
    assert (BrainScoreModel.__init__.__doc__ or '').strip()
    assert (BrainScoreModel.process.__doc__ or '').strip()


def test_positional_capability_tail_warns():
    with pytest.warns(DeprecationWarning, match="positionally"):
        BrainScoreModel('id', None, {}, {}, None, 8, 'layer.10')  # 7th positional = footgun


def test_keyword_calls_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        BrainScoreModel('id', None, {}, {}, None, action_fn=_f)  # must not raise
