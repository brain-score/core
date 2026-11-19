"""The step rule of a rating scale. The grid of a scale is decimal: the points
Decimal(repr(low)) + n * Decimal(repr(step)). A value is on the step when its decimal index
is within 1e-9 of an integer n, or when it is the float of the grid point n. One rule
serves the reply parse and the stored check."""
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
    (1.1 + 12 * 0.1, 1.1, 7.1, 0.1, True),                      # computed in float
    (0.1, -1e15, 1, 0.1, True),                                 # review S4-02
    (1000000000000000.5, 0, 1000000000000001, 1, False),        # review S4-03
    (1e14 + 0.0625, 0, 1e15, 1, False),                         # review S4-03
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


@pytest.mark.parametrize('reply', ['not 3, 5', '3 or 4', '2.5 then 3'])
def test_a_reply_with_several_numbers_is_invalid(reply):
    value, reason = Scale(1, 5, 0.5).parse(reply)
    assert value is None and reason


@pytest.mark.parametrize('reply, value', [('4', 4.0), ('4, yes 4', 4.0), ('3 (3.0)', 3.0)])
def test_a_reply_with_one_number_is_valid_also_when_repeated(reply, value):
    assert Scale(1, 5, 1).parse(reply) == (value, None)


# Each row of the rating examples: a reply is read as complete numeric tokens with
# their exact decimal values; a notation the parser does not support is invalid.
WIDE = Scale(-10, 2000)


@pytest.mark.parametrize('reply, value', [
    ('1e3', 1000.0), ('1e1', 10.0), ('.5', 0.5), ('−3', -3.0), ('-3', -3.0),
    ('4, yes 4', 4.0), ('3 (3.0)', 3.0), ('4 points', 4.0), ('4 m2', 4.0),
    ('1000 or 1e3', 1000.0), ('The answer is 7.', 7.0),
])
def test_one_number_in_any_supported_spelling_is_that_number(reply, value):
    assert WIDE.parse(reply) == (value, None)


@pytest.mark.parametrize('reply', ['3-4', '3 or 4', '3 and 3.0000000000000001'])
def test_distinct_exact_values_are_several_numbers(reply):
    value, reason = WIDE.parse(reply)
    assert value is None and 'several numbers' in reason


@pytest.mark.parametrize('reply', ['1,000', '3,5'])
def test_an_unsupported_notation_is_invalid_never_another_number(reply):
    value, reason = WIDE.parse(reply)
    assert value is None and reply in reason
