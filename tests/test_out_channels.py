"""C7c: a subject with an action_fn advertises 'motor' in out_channels, so
compatibility pre-flight can reject an embodied benchmark on a model that has
no action_fn instead of failing at the first tick."""
from brainscore_core import BrainScoreModel


def test_motor_advertised_when_action_fn_present():
    m = BrainScoreModel('agent', None, {}, {}, None, action_fn=lambda step: None)
    assert 'motor' in m.out_channels


def test_motor_absent_without_action_fn():
    m = BrainScoreModel('feat', None, {'IT': 'layer4'}, {})
    assert 'motor' not in m.out_channels
    assert 'neural:IT' in m.out_channels


def test_behavior_still_advertised_with_readout():
    m = BrainScoreModel('clf', None, {'IT': 'layer4'}, {},
                        behavioral_readout_layer='layer4')
    assert 'behavior' in m.out_channels
