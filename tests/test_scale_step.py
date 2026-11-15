"""The step rule of a rating scale: a value is on the step when it is the float nearest
to a grid point ``low + n * step``. One rule serves the reply parse and the stored check."""
import pandas as pd
import pytest

from brainscore_core.contract import TaskContext
from brainscore_core.responses import Scale, _on_step
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

# (value, low, high, step, on the step)
TABLE = [
    (3.0, 1, 7, 1, True),
    (1.5, 1, 7, 1, False),
    (0.30000000000000004, 0.1, 0.3, 0.1, True),
    (0.3, 0.1, 0.3, 0.1, True),
    (0.25, 0.1, 0.3, 0.1, False),
    (2.3, 1.1, 7.1, 0.1, True),
    (2.3000000000000003, 1.1, 7.1, 0.1, True),
    (2.35, 1.1, 7.1, 0.1, False),
    (100000000.3, 100000000.0, 100000001.0, 0.1, True),
    (100000000.35, 100000000.0, 100000001.0, 0.1, False),
    (1e12 + 3.0, 1e12, 1e12 + 10, 1, True),
    (1e12 + 1.5, 1e12, 1e12 + 10, 1, False),
]


@pytest.mark.parametrize('value, low, high, step, on', TABLE)
def test_the_step_rule(value, low, high, step, on):
    assert _on_step(value, low, step) is on


@pytest.mark.parametrize('value, low, high, step, on', TABLE)
def test_the_reply_parse_and_the_stored_check_agree(value, low, high, step, on):
    scale = Scale(low, high, step)
    valid = on and low <= value <= high          # the bounds are exact
    parsed, error = scale.parse(repr(value))
    assert (error is None) is valid
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['s0'], 'text': ['x']}))
    stimuli.identifier = 'scale-step'
    session = stimulus_session(stimuli, ['behavior:rating'], TaskContext(response=scale))
    session.respond(StreamEvent('behavior:rating', value, None, {'stimulus_id': 's0'}))
    stored = session.collect('behavior:rating')
    assert stored['response_state'].values.tolist() == ['valid' if valid else 'invalid']


def test_the_bounds_stay_inclusive_and_exact():
    assert Scale(0.1, 0.3, 0.1).parse('0.30000000000000004')[1] == '0.30000000000000004 is outside 0.1 to 0.3'
    assert Scale(1, 7, 1).parse('7')[1] is None and Scale(1, 7, 1).parse('1')[1] is None
