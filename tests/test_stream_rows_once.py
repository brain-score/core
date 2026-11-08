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
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brainscore_core import BrainScoreModel, LayerSelector
from brainscore_core._sessions.buffered import StimulusSetSession
from brainscore_core._sessions.payloads import _stimulus_events, _table_row_position
from brainscore_core._sessions.perception import StreamingStimulusSetSession, WindowedStreamSession
from brainscore_core.streaming import BufferedSession, StreamEvent
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit

IDS = ['a', 'b']


def table(two=True, ids=IDS, index=None):
    data = {'stimulus_id': list(ids), 'sentence': [f'{i}-words' for i in ids]}
    if two:
        data['text'] = [f'{i}-text' for i in ids]
    return StimulusSet(pd.DataFrame(data, index=index))


def text_extractor(calls):
    """Output values depend on the stimulus, so a wrong row shows up."""
    def extract(stimuli, layers):
        calls.append(list(stimuli['stimulus_id']))
        column = 'sentence' if 'sentence' in stimuli.columns else 'text'
        values = np.array([[float(sum(map(ord, s))), float(len(s))] for s in stimuli[column]],
                          dtype=np.float32)
        return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
            'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
            'neuroid_id': ('neuroid', [f'{layers[0]}.0', f'{layers[0]}.1']),
            'layer': ('neuroid', [layers[0]] * 2)})
    return extract


def text_model(calls):
    return BrainScoreModel('rows-once', extractor={'text': text_extractor(calls)},
                           region_layer_map={'A': LayerSelector('late', modality='text')})


def streaming_session(kind, stimuli, record='A'):
    if kind == 'streaming':
        return StreamingStimulusSetSession(stimuli, record=record)
    if kind == 'buffered':
        return BufferedSession(stimulus_set=stimuli, streaming=True,
                               requested_output_channels=[f'neural:{record}'])
    session = StimulusSetSession(stimuli, record=record)    # compatibility class
    session.streaming = True
    return session


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

@pytest.mark.parametrize('kind', KINDS)
def test_two_column_table_gives_one_output_per_row(kind):
    calls = []
    session = streaming_session(kind, table())
    text_model(calls).participate(session)
    assert calls == [['a'], ['b']]
    assert presented(session) == [['a'], ['b']]
    assert [event.meta['stream_index'] for event in session.emitted] == [0, 1]


@pytest.mark.parametrize('kind', KINDS)
def test_streamed_rows_equal_the_batch_rows(kind):
    calls = []
    model = text_model(calls)
    batch = BufferedSession(stimulus_set=table(), requested_output_channels=['neural:A'])
    model.participate(batch)
    stream = streaming_session(kind, table())
    model.participate(stream)
    np.testing.assert_array_equal(stream.collect('neural:A').values,
                                  batch.collect('neural:A').values)


def test_batch_session_with_two_columns_is_unchanged():
    calls = []
    session = BufferedSession(stimulus_set=table(), requested_output_channels=['neural:A'])
    text_model(calls).participate(session)
    assert calls == [['a', 'b']]
    assert len(session.emitted) == 1


def test_streaming_session_still_holds_at_most_one_row():
    session = StreamingStimulusSetSession(table(ids=list('abcdef')), record='A')
    text_model([]).participate(session)
    assert session.max_events_held <= 2           # one row's two input columns


def test_two_extractors_record_one_row_once_per_channel():
    """Video and audio columns with separate extractors (the reported case)."""
    calls = []

    def extractor(name, column):
        def extract(stimuli, layers):
            calls.append((name, list(stimuli['stimulus_id'])))
            values = np.array([[float(len(str(v)))] for v in stimuli[column]], dtype=np.float32)
            return NeuroidAssembly(values, dims=['presentation', 'neuroid'], coords={
                'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                'neuroid_id': ('neuroid', [f'{name}.0']), 'layer': ('neuroid', [layers[0]])})
        return extract

    model = BrainScoreModel('av', extractor={'vision': extractor('v', 'image_path'),
                                             'audio': extractor('a', 'audio_path')},
                            region_layer_map={'IT': LayerSelector('late', modality='vision'),
                                              'A1': LayerSelector('mid', modality='audio')})
    stimuli = StimulusSet({'stimulus_id': IDS, 'image_path': ['a.png', 'bb.png'],
                           'audio_path': ['a.wav', 'bbb.wav']})
    session = StreamingStimulusSetSession(stimuli, record=['IT', 'A1'])
    model.participate(session)
    assert sorted(calls) == [('a', ['a']), ('a', ['b']), ('v', ['a']), ('v', ['b'])]
    assert presented(session, 'neural:IT') == [['a'], ['b']]
    assert presented(session, 'neural:A1') == [['a'], ['b']]


def test_view_that_drops_a_stream_still_presents_each_row_once():
    """A text-only subject on a text + image table never sees the image events."""
    calls = []
    stimuli = StimulusSet({'stimulus_id': IDS, 'image_path': ['a.png', 'b.png'],
                           'sentence': ['a-words', 'b-words']})
    session = StreamingStimulusSetSession(stimuli, record='A')
    text_model(calls).participate(session)
    assert calls == [['a'], ['b']]


def test_repeated_index_labels_keep_distinct_rows():
    """v5 defect: distinct stimuli that share an index label are not merged."""
    calls = []
    session = StreamingStimulusSetSession(table(ids=['a', 'b', 'c'], index=[7, 7, 8]), record='A')
    text_model(calls).participate(session)
    assert calls == [['a'], ['b'], ['c']]


def test_repeated_stimulus_ids_in_the_table_stay_separate_presentations():
    calls = []
    session = StreamingStimulusSetSession(table(ids=['a', 'a']), record='A')
    text_model(calls).participate(session)
    assert calls == [['a'], ['a']]


def test_row_cut_short_is_presented_and_never_dropped():
    events = [e for e in _stimulus_events(table()) if not (e.meta['stimulus_id'] == 'a'
                                                          and e.meta['column'] == 'text')]
    calls = []
    session = BufferedSession(events, streaming=True, requested_output_channels=['neural:A'])
    session.stimulus_set = table()
    text_model(calls).participate(session)
    assert calls == [['a'], ['b']]


# --- one input column: unchanged --------------------------------------------

@pytest.mark.parametrize('kind', KINDS)
def test_one_column_table_is_unchanged(kind):
    calls = []
    session = streaming_session(kind, table(two=False))
    text_model(calls).participate(session)
    assert calls == [['a'], ['b']]
    assert presented(session) == [['a'], ['b']]


@pytest.mark.parametrize('producer', ['buffered', 'streaming'])
def test_one_column_events_carry_no_new_metadata(producer):
    stimuli = table(two=False)
    events = (_stimulus_events(stimuli) if producer == 'buffered'
              else drain(StreamingStimulusSetSession(stimuli, record='A')))
    assert [event.meta for event in events] == [
        {'stimulus_id': 'a', 'stimulus_index': 0, 'column': 'sentence'},
        {'stimulus_id': 'b', 'stimulus_index': 1, 'column': 'sentence'}]


@pytest.mark.parametrize('producer', ['buffered', 'streaming'])
def test_two_column_events_mark_their_row_position(producer):
    stimuli = table(index=[10, 10])
    events = (_stimulus_events(stimuli) if producer == 'buffered'
              else drain(StreamingStimulusSetSession(stimuli, record='A')))
    assert [(e.meta['column'], e.meta['stimulus_index'], e.meta['table_row']) for e in events] == [
        ('sentence', 10, 0), ('text', 10, 0), ('sentence', 10, 1), ('text', 10, 1)]


# --- external events are never merged -----------------------------------------

def external(meta_list, channel='text'):
    return [StreamEvent(channel, f'payload-{i}', 0.0, meta=dict(meta)) for i, meta in enumerate(meta_list)]


def run_external(events, stimuli=None):
    calls = []
    session = BufferedSession(events, streaming=True, requested_output_channels=['neural:A'])
    if stimuli is not None:
        session.stimulus_set = stimuli
    text_model(calls).participate(session)
    return calls


def test_external_events_with_repeated_hints_are_never_merged():
    meta = {'stimulus_id': 'a', 'stimulus_index': 0}
    events = external([{**meta, 'column': 'sentence'}, {**meta, 'column': 'text'}])
    assert len(run_external(events, table())) == 2


def test_external_events_without_hints_are_never_merged():
    """Only a stimulus id (repeated), no index and no row marker."""
    events = [StreamEvent('text', word, 0.0, meta={'stimulus_id': 'same'}) for word in ('one', 'two')]
    assert len(run_external(events)) == 2
    assert len(run_external(deepcopy(events), table())) == 2


@pytest.mark.parametrize('change', ['tuple_marker', 'bool_marker', 'out_of_range', 'no_index',
                                    'wrong_index', 'no_id', 'wrong_id', 'foreign_column',
                                    'wrong_channel', 'no_table'])
def test_external_marker_that_does_not_match_the_table_is_never_merged(change):
    events = _stimulus_events(table())[:2]                # row 'a': sentence, text
    stimuli = None if change == 'no_table' else table()
    for event in events:
        meta = event.meta
        if change == 'tuple_marker':
            meta['table_row'] = ('forged', 0)
        elif change == 'bool_marker':
            meta['table_row'] = False
        elif change == 'out_of_range':
            meta['table_row'] = 5
        elif change == 'no_index':
            del meta['stimulus_index']
        elif change == 'wrong_index':
            meta['stimulus_index'] = 1
        elif change == 'no_id':
            del meta['stimulus_id']
        elif change == 'wrong_id':
            meta['stimulus_id'] = 'b'
        elif change == 'foreign_column':
            meta['column'] = 'image_path'
    if change == 'wrong_channel':
        events = [StreamEvent('vision', e.payload, e.t_ms, meta=e.meta) for e in events]
        session = BufferedSession(events, streaming=True, requested_output_channels=['neural:A'])
        session.stimulus_set = table()
        assert all(_table_row_position(session, e) is None for e in events)
        return
    assert len(run_external(events, stimuli)) == 2


def test_repeated_column_of_one_row_is_a_second_presentation():
    """Showing a stimulus twice is a repetition, not two columns of one row."""
    first = _stimulus_events(table())[0]
    assert len(run_external([first, deepcopy(first)], table())) == 2


def test_events_of_one_row_at_different_times_are_separate():
    events = _stimulus_events(table())[:2]
    events[1].t_ms = 50.0
    assert len(run_external(events, table())) == 2


def test_table_row_position_reads_only_the_event_and_the_table():
    session = BufferedSession()
    session.stimulus_set = table()
    event = _stimulus_events(table())[0]
    assert _table_row_position(session, event) == 0
    session.settings = {'_table_rows': 'anything'}
    assert _table_row_position(session, event) == 0


# --- settings replaced before and after init ------------------------------------

@pytest.mark.parametrize('kind', ['buffered', 'streaming'])
def test_settings_installed_before_init_keep_one_output_per_row(kind):
    original = {'number_of_trials': 3, '_table_rows': 'user-token'}
    if kind == 'buffered':
        class Source(BufferedSession):
            def __init__(self):
                self.settings = original
                super().__init__(stimulus_set=table(), streaming=True,
                                 requested_output_channels=['neural:A'])
    else:
        class Source(StreamingStimulusSetSession):
            def __init__(self):
                self.settings = original
                super().__init__(table(), record='A')
    calls = []
    session = Source()
    text_model(calls).participate(session)
    assert calls == [['a'], ['b']]
    assert session.settings == {'number_of_trials': 3, '_table_rows': 'user-token'}


@pytest.mark.parametrize('kind', ['buffered', 'streaming'])
@pytest.mark.parametrize('settings', [{'number_of_trials': 3}, {'_table_rows': 'user-token'}, None])
def test_settings_replaced_after_init_keep_one_output_per_row(kind, settings):
    calls = []
    session = streaming_session(kind, table())
    session.settings = deepcopy(settings)
    text_model(calls).participate(session)
    assert calls == [['a'], ['b']]


# --- nested and replayed sessions ----------------------------------------------

def test_nested_session_groups_its_own_rows_only():
    seen, inner_outputs = [], []
    inner_calls = []

    def extract(stimuli, layers):
        seen.extend(stimuli['stimulus_id'])
        if stimuli['stimulus_id'].iloc[0] == 'a' and not inner_outputs:
            inner = StreamingStimulusSetSession(table(ids=['inner']), record='A')
            text_model(inner_calls).participate(inner)
            inner_outputs.append(presented(inner))
        return text_extractor([])(stimuli, layers)

    model = BrainScoreModel('outer', extractor={'text': extract},
                            region_layer_map={'A': LayerSelector('late', modality='text')})
    session = StreamingStimulusSetSession(table(), record='A')
    model.participate(session)
    assert seen == ['a', 'b']
    assert inner_outputs == [[['inner']]]
    assert presented(session) == [['a'], ['b']]


def replay(events, source_session, settings=True):
    """What brainscore.experiments.replay_sessions builds from a record."""
    session = BufferedSession(deepcopy(events))
    session.stimulus_set = deepcopy(source_session.stimulus_set)
    session.streaming = True
    session.requested_output_channels = ('neural:A',)
    if settings:
        session.settings = deepcopy(getattr(source_session, 'settings', None))
    return session


def test_replayed_new_record_presents_each_row_once_every_time():
    source = StreamingStimulusSetSession(table(), record='A')
    events = drain(source)
    model = text_model(calls := [])
    counts = []
    for _ in range(2):
        session = replay(events, source)
        model.participate(session)
        counts.append(len(session.emitted))
    assert counts == [2, 2]
    assert calls == [['a'], ['b'], ['a'], ['b']]


def test_replayed_base_record_keeps_base_outputs():
    """A base record has no row marker: it replays exactly as base ran it."""
    source = StreamingStimulusSetSession(table(), record='A')
    base_events = drain(source)
    for event in base_events:
        del event.meta['table_row']
    calls = []
    session = replay(base_events, source)
    text_model(calls).participate(session)
    assert calls == [['a'], ['a'], ['b'], ['b']]


# --- complete stream windows are unchanged ---------------------------------------

def test_stream_windows_are_unchanged():
    calls = []

    def extract(stimuli, layers):
        calls.append(list(stimuli['stimulus_id']))
        return NeuroidAssembly(np.ones((len(stimuli), 1), dtype=np.float32), dims=['presentation', 'neuroid'],
                               coords={'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                                       'neuroid_id': ('neuroid', ['n0']), 'layer': ('neuroid', [layers[0]])})

    model = BrainScoreModel('windows', extractor={'vision': extract},
                            region_layer_map={'IT': LayerSelector('late', modality='vision')})
    session = WindowedStreamSession([f'f{i}.png' for i in range(6)], fps=10.0, window_ms=200.0, record='IT')
    model.participate(session)
    assert calls == [['w0_f0', 'w0_f1'], ['w200_f0', 'w200_f1'], ['w400_f0', 'w400_f1']]
    assert [event.meta['stream_index'] for event in session.emitted] == [0, 1, 2]


# --- ported Codex probes (v5, v6, v7) -------------------------------------------

@pytest.mark.parametrize('positions', [[0, 0], [0, 1]])
def test_v5_external_stream_positions_do_not_erase_distinct_presentations(positions):
    calls = []

    def extract(stimuli, layers):
        calls.append(list(stimuli['stimulus_id']))
        return NeuroidAssembly(np.ones((len(stimuli), 1), dtype=np.float32), dims=['presentation', 'neuroid'],
                               coords={'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                                       'neuroid_id': ('neuroid', ['n']), 'layer': ('neuroid', [layers[0]])})

    model = BrainScoreModel('ext', extractor={'vision': extract},
                            region_layer_map={'IT': LayerSelector('late', modality='vision')})
    events = [StreamEvent('vision:still', Path(sid + '.png'), float(i),
                          meta={'column': 'image_path', 'stimulus_id': sid, 'stimulus_position': pos,
                                'episode_id': i, 'table_row': pos})
              for i, (sid, pos) in enumerate(zip(IDS, positions))]
    session = BufferedSession(events, streaming=True, requested_output_channels=['neural:IT'])
    model.participate(session)
    assert len(calls) == 2
    assert session.collect('neural:IT').sizes['presentation'] == 2


@pytest.mark.parametrize('matching', [False, True])
def test_v6_external_marker_without_table_does_not_discard_events(matching):
    events = external([{'column': 'sentence', 'stimulus_id': sid, 'stimulus_index': 0,
                        'table_row': 0 if matching else ('foreign', 0)} for sid in IDS])
    for event, sid in zip(events, IDS):
        event.payload = sid
    assert len(run_external(events)) == 2


def test_v6_independent_sessions_with_equal_markers_stay_local():
    calls = []
    model = text_model(calls)
    for ids in (['a', 'b'], ['c', 'd']):
        session = StreamingStimulusSetSession(table(ids=ids), record='A')
        model.participate(session)
        assert len(session.emitted) == 2
    assert calls == [['a'], ['b'], ['c'], ['d']]


@pytest.mark.parametrize('hint', ['foreign', 'range', 'index', 'id', 'no_index', 'no_table', 'valid'])
def test_v7_external_hint_requires_complete_matching_identity(hint):
    session = BufferedSession()
    if hint != 'no_table':
        session.stimulus_set = table()
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


def test_v7_missing_index_cannot_collapse_external_repeated_id_events():
    events = [StreamEvent('text', word, float(i), meta={'column': 'sentence', 'stimulus_id': 'same',
                                                        'table_row': 0})
              for i, word in enumerate(['first', 'second'])]
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['same', 'same'], 'sentence': ['first', 'second']},
                                       index=[3, 4]))
    assert len(run_external(events, stimuli)) == 2


# --- null index labels and the catalog (Codex PR B review) ------------------------

@pytest.mark.parametrize('label', [np.nan, pd.NA], ids=['NaN', 'pdNA'])
@pytest.mark.parametrize('kind', KINDS)
def test_null_index_label_presents_each_row_once(kind, label):
    calls = []
    session = streaming_session(kind, table(index=[label, 2]))
    text_model(calls).participate(session)
    assert calls == [['a'], ['b']]


@pytest.mark.parametrize('label', [np.nan, pd.NA], ids=['NaN', 'pdNA'])
def test_null_index_label_is_matched_only_when_present(label):
    session = BufferedSession()
    session.stimulus_set = table(index=[label, 2])
    event = _stimulus_events(session.stimulus_set)[0]
    assert _table_row_position(session, event) == 0
    event.meta['stimulus_index'] = 2                 # a value never equals a null label
    assert _table_row_position(session, event) is None
    del event.meta['stimulus_index']                 # an absent label stays invalid
    assert _table_row_position(session, event) is None
    other = _stimulus_events(session.stimulus_set)[2]
    other.meta['stimulus_index'] = label             # a null never equals a value label
    assert _table_row_position(session, other) is None


@pytest.mark.parametrize('channel', ['vision', 'text', 'audio', 'video'])
def test_catalog_declares_the_row_metadata(channel):
    from brainscore_core import io_catalog
    keys = io_catalog.get(channel).meta_keys
    assert {'stimulus_id', 'stimulus_index', 'column', 'table_row'} <= set(keys)
    assert 'table_row' in io_catalog.describe(channel)


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
