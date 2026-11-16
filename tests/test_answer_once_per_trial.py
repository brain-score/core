"""Rule 5.12: a second answer for one trial is an error, whatever happened on earlier trials."""
import numpy as np
import pandas as pd
import pytest

from brainscore_core._sessions.buffered import stimulus_session
from brainscore_core.streaming import StreamEvent


def _session(channels=('behavior',), rows=3):
    table = pd.DataFrame({'stimulus_id': [f's{i}' for i in range(rows)],
                          'image_path': [f'p{i}.png' for i in range(rows)]})
    return stimulus_session(table, channels)


def _read_trial(session):
    """Read events up to and including the next stimulus event; return it."""
    while (event := session.next_input()) is not None:
        if event.channel.startswith('vision'):
            return event


def _answer(session, channel='behavior', value=0, **meta):
    """Answer with a payload that names ``value``: a label ``a<value>`` or a neural row of it."""
    payload = np.full(4, float(value)) if channel.startswith('neural') else f'a{value}'
    session.respond(StreamEvent(channel, payload, None, meta))


def _answers(session, channel='behavior'):
    """The collected answers, one per row in the order the session keeps them."""
    collected = session.collect(channel)
    if channel.startswith('neural'):
        return [float(row[0]) for row in collected.values]
    return [str(collected['choice'].values[row.argmax()]) for row in collected.values]


def test_skipped_trials_do_not_leave_room_for_extra_answers():
    session = _session()
    _read_trial(session), _read_trial(session), _read_trial(session)
    _answer(session)
    with pytest.raises(ValueError, match="'behavior'.*twice.*trial 3"):
        _answer(session)


def test_one_answer_per_trial_for_each_of_two_channels():
    session = _session(('behavior', 'neural:IT'), rows=2)
    for row in range(2):
        _read_trial(session)
        _answer(session, value=row)
        _answer(session, 'neural:IT', value=row)
    assert _answers(session) == ['a0', 'a1']
    assert _answers(session, 'neural:IT') == [0.0, 1.0]


def test_delayed_answers_attributed_by_row_identity():
    session = _session()
    events = [_read_trial(session) for _ in range(3)]
    for event in events:
        _answer(session, value=event.meta['stimulus_index'], stimulus_index=event.meta['stimulus_index'])
    assert _answers(session) == ['a0', 'a1', 'a2']


def test_answers_out_of_row_order_keep_the_value_of_their_stimulus():
    """Answers for rows 2, 0, 1: each collected answer keeps the value of its own stimulus
    (the pair of id and value is checked, not the position), and a second answer for a
    row that has one is refused."""
    session = _session(('behavior', 'neural:IT'))
    events = [_read_trial(session) for _ in range(3)]
    for row in (2, 0, 1):
        meta = {'stimulus_index': events[row].meta['stimulus_index'],
                'stimulus_id': events[row].meta['stimulus_id']}
        _answer(session, value=row, **meta)
        _answer(session, 'neural:IT', value=row, **meta)
    behavior, neural = session.collect('behavior'), session.collect('neural:IT')
    assert dict(zip(behavior['stimulus_id'].values, _answers(session))) == {'s0': 'a0', 's1': 'a1', 's2': 'a2'}
    assert dict(zip(neural['stimulus_id'].values, _answers(session, 'neural:IT'))) == \
        {'s0': 0.0, 's1': 1.0, 's2': 2.0}
    with pytest.raises(ValueError, match="twice"):
        _answer(session, value=0, stimulus_index=events[0].meta['stimulus_index'],
                stimulus_id=events[0].meta['stimulus_id'])


def test_delayed_duplicate_for_one_row_is_an_error():
    session = _session()
    events = [_read_trial(session) for _ in range(3)]
    _answer(session, stimulus_index=events[0].meta['stimulus_index'])
    _answer(session, stimulus_index=events[1].meta['stimulus_index'])
    with pytest.raises(ValueError, match="twice.*trial 1"):
        _answer(session, stimulus_index=events[0].meta['stimulus_index'])


def test_delayed_answers_without_identity_go_in_row_order_then_error():
    session = _session()
    while session.next_input() is not None:
        pass
    for _ in range(3):
        _answer(session)
    with pytest.raises(ValueError, match="twice.*trial 3"):
        _answer(session)


def _repeated(labels=(0, 1, 0, 1)):
    table = pd.DataFrame({'stimulus_id': [f's{i}' for i in range(len(labels))],
                          'image_path': [f'p{i}.png' for i in range(len(labels))]},
                         index=list(labels))
    return stimulus_session(table, ('behavior', 'neural:IT'))


def _answer_copying(session, event):
    """Answer on both channels with the event's meta and a payload that names its row."""
    row = int(event.meta['stimulus_id'][1:])
    for channel in ('behavior', 'neural:IT'):
        payload = np.full(2, float(row)) if channel.startswith('neural') else f'a{row}'
        session.respond(StreamEvent(channel, payload, event.t_ms, dict(event.meta)))


def _one_answer_per_row_in_order(session, rows=4):
    ids = [f's{i}' for i in range(rows)]
    for channel in ('behavior', 'neural:IT'):
        assert list(session.collect(channel)['stimulus_id'].values) == ids, channel
    assert _answers(session) == [f'a{i}' for i in range(rows)]
    assert _answers(session, 'neural:IT') == [float(i) for i in range(rows)]


def test_repeated_labels_one_answer_per_row_in_order():
    session = _repeated()
    for _ in range(4):
        _answer_copying(session, _read_trial(session))
    _one_answer_per_row_in_order(session)


def test_repeated_labels_one_answer_per_row_after_all_rows_read():
    session = _repeated()
    events = [_read_trial(session) for _ in range(4)]
    for event in events:
        _answer_copying(session, event)
    _one_answer_per_row_in_order(session)


def test_repeated_labels_extra_answer_is_an_error():
    session = _repeated()
    events = [_read_trial(session) for _ in range(4)]
    _answer(session, stimulus_index=0)
    _answer(session, stimulus_index=0)
    with pytest.raises(ValueError, match="twice"):
        _answer(session, stimulus_index=0)


def test_many_rows_with_repeated_labels_are_answered_once_each():
    # A label shared by many rows takes one answer per row, then an error.
    labels = [i % 50 for i in range(2000)]
    session = _repeated(labels)
    session.outputs = ('behavior',)
    events = [_read_trial(session) for _ in labels]
    for event in events:
        _answer(session, stimulus_index=event.meta['stimulus_index'])
    assert len(session.emitted) == 2000
    assert session._row_trials[7] == [8 + 50 * k for k in range(40)]
    with pytest.raises(ValueError, match="twice"):
        _answer(session, stimulus_index=7)
