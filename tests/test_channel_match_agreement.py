"""Pre-flight and the session input filter use one match rule: exact channel strings."""

import random
from types import SimpleNamespace

import pytest

from brainscore_core._evaluation.compatibility import check_channel_compatibility
from brainscore_core._evaluation.compatibility import CompatibilityError
from brainscore_core.streaming import Session, StreamEvent

POOL = ['vision:image', 'vision:video', 'text', 'audio']


def _delivered(inputs, declared):
    view = Session([StreamEvent(channel, 'x', 0.0) for channel in sorted(inputs)],
                   ['neural:IT']).declare_inputs(declared)
    seen = set()
    while (event := view.next_input()) is not None:
        seen.add(event.channel)
    return seen


def test_consumed_inputs_equal_delivered_inputs():
    rng = random.Random(0)
    for _ in range(300):
        declared = set(rng.sample(POOL, rng.randint(1, len(POOL))))
        inputs = set(rng.sample(POOL, rng.randint(1, len(POOL))))
        subject = SimpleNamespace(identifier='s', in_channels=declared, out_channels={'neural:IT'},
                                  required_channels=set(), traits=set(), output_inputs={})
        benchmark = SimpleNamespace(identifier='b', required_input_channels=inputs,
                                    requested_output_channels={'neural:IT'})
        delivered = _delivered(inputs, declared)
        try:
            match = check_channel_compatibility(subject, benchmark)
        except CompatibilityError:
            assert delivered == set()
            continue
        assert set(match.consumed_inputs) == delivered
        assert set(match.dropped_inputs) == inputs - delivered


def test_one_form_does_not_stand_for_another():
    assert _delivered({'vision:image'}, {'vision:video'}) == set()
    assert _delivered({'vision:video'}, {'vision:image'}) == set()
    assert _delivered({'vision:image'}, {'vision:image', 'vision:video'}) == {'vision:image'}
