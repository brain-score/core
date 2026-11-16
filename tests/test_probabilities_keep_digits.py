"""behavior:probabilities keeps the digits the subject gave: float64 stays float64."""

import numpy as np
import pandas as pd
import pytest

from brainscore_core.contract import TaskContext
from brainscore_core.response_formats import response_record
from brainscore_core.responses import Choice
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import stimulus_session
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

CHANNEL = 'behavior:probabilities'
# Rows whose float64 digits a float32 copy loses.
ROWS = [[0.1234567890123456, 0.8765432109876544], [1 / 3, 2 / 3]]


def _session():
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['s0', 's1'], 'text': ['x', 'y']}))
    stimuli.identifier = 'probabilities-keep-digits'
    session = stimulus_session(stimuli, [CHANNEL], TaskContext(response=Choice(['cat', 'dog'])))
    while session.next_input() is not None:
        pass
    return session


def _assembly(values, ids=('s0', 's1')):
    return BehavioralAssembly(values, dims=['presentation', 'choice'],
                              coords={'stimulus_id': ('presentation', list(ids)),
                                      'choice': ['cat', 'dog']},
                              attrs={'response': response_record(CHANNEL, 'readout')})


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_an_assembly_is_collected_in_the_float_type_and_digits_it_was_given(dtype):
    given = np.array(ROWS, dtype=dtype)
    session = _session()
    session.respond(StreamEvent(CHANNEL, _assembly(given), None, {}))
    answer = session.collect(CHANNEL)
    assert answer.values.dtype == dtype
    assert answer.values.tobytes() == given.tobytes()
    assert set(answer['response_state'].values) == {'valid'}


@pytest.mark.parametrize('given, kept', [(np.float64, np.float64), (np.float32, np.float32),
                                         (np.int64, np.float32)])
def test_a_cell_keeps_float64_and_stores_any_other_number_type_as_float32(given, kept):
    session = _session()
    rows = [np.array(row, dtype=given) if given != np.int64 else np.array([0, 1], dtype=given)
            for row in ROWS]
    for stimulus_id, row in zip(['s0', 's1'], rows):
        session.respond(StreamEvent(CHANNEL, row, None, {'stimulus_id': stimulus_id}))
    answer = session.collect(CHANNEL)
    assert answer.values.dtype == kept
    assert answer.values.tobytes() == np.array(rows, dtype=kept).tobytes()


def test_rows_of_both_float_types_on_one_channel_keep_each_row_exactly():
    session = _session()
    session.respond(StreamEvent(CHANNEL, _assembly(np.array(ROWS[:1], dtype=np.float32), ['s0']),
                                None, {}))
    session.respond(StreamEvent(CHANNEL, _assembly(np.array(ROWS[1:], dtype=np.float64), ['s1']),
                                None, {}))
    answer = session.collect(CHANNEL)
    # float32 to float64 is exact, so the collected rows are the given rows.
    assert answer.values[0].tolist() == np.array(ROWS[0], dtype=np.float32).tolist()
    assert answer.values[1].tolist() == ROWS[1]


def test_a_float_type_other_than_float32_or_float64_is_refused():
    session = _session()
    with pytest.raises(ValueError, match='float16'):
        session.respond(StreamEvent(CHANNEL, _assembly(np.array(ROWS, dtype=np.float16)), None, {}))


@pytest.mark.parametrize('byte_order', ['<f8', '>f8'])
def test_a_float64_in_either_byte_order_keeps_its_digits_as_a_cell_and_as_an_assembly(byte_order):
    given = np.array(ROWS, dtype=byte_order)
    cells = _session()
    for stimulus_id, row in zip(['s0', 's1'], given):
        cells.respond(StreamEvent(CHANNEL, row, None, {'stimulus_id': stimulus_id}))
    whole = _session()
    whole.respond(StreamEvent(CHANNEL, _assembly(given), None, {}))
    for session in (cells, whole):
        answer = session.collect(CHANNEL)
        assert answer.values.dtype == np.float64
        assert answer.values.tolist() == ROWS
