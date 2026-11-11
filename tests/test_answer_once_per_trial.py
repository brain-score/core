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


def _answer(session, channel='behavior', **meta):
    payload = np.zeros(4) if channel.startswith('neural') else 'x'
    session.respond(StreamEvent(channel, payload, None, meta))


def test_skipped_trials_do_not_leave_room_for_extra_answers():
    session = _session()
    _read_trial(session), _read_trial(session), _read_trial(session)
    _answer(session)
    with pytest.raises(ValueError, match="'behavior'.*twice.*trial 3"):
        _answer(session)


def test_one_answer_per_trial_for_each_of_two_channels():
    session = _session(('behavior', 'neural:IT'), rows=2)
    for _ in range(2):
        _read_trial(session)
        _answer(session)
        _answer(session, 'neural:IT')


def test_delayed_answers_attributed_by_row_identity():
    session = _session()
    events = [_read_trial(session) for _ in range(3)]
    for event in events:
        _answer(session, stimulus_index=event.meta['stimulus_index'])


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
    for channel in ('behavior', 'neural:IT'):
        payload = np.zeros(2) if channel.startswith('neural') else 'x'
        session.respond(StreamEvent(channel, payload, event.t_ms, dict(event.meta)))


def test_repeated_labels_one_answer_per_row_in_order():
    session = _repeated()
    for _ in range(4):
        _answer_copying(session, _read_trial(session))


def test_repeated_labels_one_answer_per_row_after_all_rows_read():
    session = _repeated()
    events = [_read_trial(session) for _ in range(4)]
    for event in events:
        _answer_copying(session, event)


def test_repeated_labels_extra_answer_is_an_error():
    session = _repeated()
    events = [_read_trial(session) for _ in range(4)]
    _answer(session, stimulus_index=0)
    _answer(session, stimulus_index=0)
    with pytest.raises(ValueError, match="twice"):
        _answer(session, stimulus_index=0)
