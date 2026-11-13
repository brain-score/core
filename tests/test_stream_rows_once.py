"""A streamed table row is presented once, however many input columns it has.

A row with two input columns (``video_path`` and ``audio_path``) reaches the
subject as one event per column. Before this fix the streaming driver showed
the full row once per event, so every stimulus was measured twice (A, A, B, B),
which breaks rule 4.5 ("each stimulus is shown once").

The table producers now mark the events of a several-column row with
``meta['table_row']`` (the row's position). The driver presents consecutive
events of one row once, and only when the marker matches the session's own
table. Every other event, including every event of a base record, is
presented alone, as before.
"""

import numpy as np
import pandas as pd
import pytest

from brainscore_core._sessions.payloads import _stimulus_events, _table_row_position
from brainscore_core._sessions.buffered import stimulus_session
from brainscore_core.streaming import Session, StreamEvent
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

IDS = ['a', 'b']


def table(two=True, ids=IDS, index=None):
    data = {'stimulus_id': list(ids), 'sentence': [f'{i}-words' for i in ids]}
    if two:
        data['text'] = [f'{i}-text' for i in ids]
    return StimulusSet(pd.DataFrame(data, index=index))


def presented(session, channel='neural:A'):
    """Stimulus ids of each emitted output, in order."""
    def ids(payload):
        coord = 'stimulus_id' if 'stimulus_id' in payload.coords else 'presentation'
        return list(payload[coord].values)
    return [ids(event.payload) for event in session.emitted if event.channel == channel]


def drain(session):
    events = []
    while (event := session.next_input()) is not None:
        events.append(event)
    return events


KINDS = ['streaming', 'buffered', 'compat']


# --- the defect: two input columns ------------------------------------------


# --- one input column: unchanged --------------------------------------------


@pytest.mark.parametrize('producer', ['buffered', 'streaming'])
def test_one_column_events_carry_no_new_metadata(producer):
    stimuli = table(two=False)
    events = (_stimulus_events(stimuli) if producer == 'buffered'
              else drain(stimulus_session(stimuli, ['neural:A'], streaming=True)))
    assert [event.meta for event in events] == [
        {'stimulus_id': 'a', 'stimulus_index': 0, 'column': 'sentence'},
        {'stimulus_id': 'b', 'stimulus_index': 1, 'column': 'sentence'}]


@pytest.mark.parametrize('producer', ['buffered', 'streaming'])
def test_two_column_events_mark_their_row_position(producer):
    stimuli = table(index=[10, 10])
    events = (_stimulus_events(stimuli) if producer == 'buffered'
              else drain(stimulus_session(stimuli, ['neural:A'], streaming=True)))
    assert [(e.meta['column'], e.meta['stimulus_index'], e.meta['table_row']) for e in events] == [
        ('sentence', 10, 0), ('text', 10, 0), ('sentence', 10, 1), ('text', 10, 1)]


# --- external events are never merged -----------------------------------------


def test_table_row_position_reads_only_the_event_and_the_table():
    session = Session(())
    session.stimuli = table()
    event = _stimulus_events(table())[0]
    assert _table_row_position(session, event) == 0
    session.settings = {'_table_rows': 'anything'}
    assert _table_row_position(session, event) == 0


# --- settings replaced before and after init ------------------------------------


# --- nested and replayed sessions ----------------------------------------------


# --- complete stream windows are unchanged ---------------------------------------


# --- ported Codex probes (v5, v6, v7) -------------------------------------------


@pytest.mark.parametrize('hint', ['foreign', 'range', 'index', 'id', 'no_index', 'no_table', 'valid'])
def test_v7_external_hint_requires_complete_matching_identity(hint):
    session = Session(())
    if hint != 'no_table':
        session.stimuli = table()
    meta = {'table_row': 0, 'stimulus_id': 'a', 'stimulus_index': 0, 'column': 'sentence'}
    if hint == 'foreign':
        meta['table_row'] = ('other', 0)
    if hint == 'range':
        meta['table_row'] = 20
    if hint == 'index':
        meta['stimulus_index'] = 1
    if hint == 'id':
        meta['stimulus_id'] = 'b'
    if hint == 'no_index':
        del meta['stimulus_index']
    actual = _table_row_position(session, StreamEvent('text', 'x', 0, meta=meta))
    assert actual == (0 if hint == 'valid' else None)


# --- null index labels and the catalog (Codex PR B review) ------------------------


@pytest.mark.parametrize('label', [np.nan, pd.NA], ids=['NaN', 'pdNA'])
def test_null_index_label_is_matched_only_when_present(label):
    session = Session(())
    session.stimuli = table(index=[label, 2])
    event = _stimulus_events(session.stimuli)[0]
    assert _table_row_position(session, event) == 0
    event.meta['stimulus_index'] = 2                 # a value never equals a null label
    assert _table_row_position(session, event) is None
    del event.meta['stimulus_index']                 # an absent label stays invalid
    assert _table_row_position(session, event) is None
    other = _stimulus_events(session.stimuli)[2]
    other.meta['stimulus_index'] = label             # a null never equals a value label
    assert _table_row_position(session, other) is None


@pytest.mark.parametrize('channel', ['vision', 'text', 'audio'])
def test_catalog_declares_the_row_metadata(channel):
    from brainscore_core import io_catalog
    keys = io_catalog.get(channel).meta_keys
    assert {'stimulus_id', 'stimulus_index', 'column', 'table_row'} <= set(keys)


@pytest.mark.parametrize('marked', [False, True])
def test_records_with_and_without_the_marker_validate(marked):
    """Meta keys are descriptive: base records (no table_row) and new ones both start."""
    import warnings
    from brainscore_core.streaming import _validate_inputs
    events = _stimulus_events(table())
    if not marked:
        for event in events:
            del event.meta['table_row']
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _validate_inputs(events)
