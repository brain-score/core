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

The driver behavior (one presentation per row, external events never merged)
is tested through sessions in unified tests/brainscore_model/test_stream_rows_once.py.
"""

import pandas as pd
import pytest

from brainscore_core._sessions.buffered import stimulus_session
from brainscore_core.streaming import Session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

IDS = ['a', 'b']


def table(two=True, ids=IDS, index=None):
    data = {'stimulus_id': list(ids), 'sentence': [f'{i}-words' for i in ids]}
    if two:
        data['text'] = [f'{i}-text' for i in ids]
    return StimulusSet(pd.DataFrame(data, index=index))


def drain(session):
    events = []
    while (event := session.next_input()) is not None:
        events.append(event)
    return events


# --- events of a table row ------------------------------------------------


@pytest.mark.parametrize('producer', ['buffered', 'streaming'])
def test_one_column_events_carry_no_new_metadata(producer):
    stimuli = table(two=False)
    events = drain(stimulus_session(stimuli, ['neural:A'], streaming=producer == 'streaming'))
    assert [event.meta for event in events] == [
        {'stimulus_id': 'a', 'stimulus_index': 0, 'column': 'sentence'},
        {'stimulus_id': 'b', 'stimulus_index': 1, 'column': 'sentence'}]


@pytest.mark.parametrize('producer', ['buffered', 'streaming'])
def test_two_column_events_mark_their_row_position(producer):
    stimuli = table(index=[10, 10])
    events = drain(stimulus_session(stimuli, ['neural:A'], streaming=producer == 'streaming'))
    assert [(e.meta['column'], e.meta['stimulus_index'], e.meta['table_row']) for e in events] == [
        ('sentence', 10, 0), ('text', 10, 0), ('sentence', 10, 1), ('text', 10, 1)]


# --- the catalog ------------------------------------------------------------


@pytest.mark.parametrize('channel', ['vision', 'text', 'audio'])
def test_catalog_declares_the_row_metadata(channel):
    from brainscore_core import io_catalog
    keys = io_catalog.get(channel).meta_keys
    assert {'stimulus_id', 'stimulus_index', 'column', 'table_row'} <= set(keys)


@pytest.mark.parametrize('marked', [False, True])
def test_records_with_and_without_the_marker_validate(marked):
    """Meta keys are descriptive: base records (no table_row) and new ones both start."""
    import warnings
    events = drain(stimulus_session(table(), ['neural:A']))
    if not marked:
        for event in events:
            del event.meta['table_row']
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        Session(events, ['neural:A'])  # a session checks its inputs when it starts (rule 3.4)
