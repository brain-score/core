"""Contract v3 rule 5.5: a stream the subject cannot take in never enters it."""

import numpy as np
import pandas as pd

from brainscore_core.streaming import (
    CONTROL_CHANNEL, Session, StreamEvent, begin_event, end_event, respond_event,
)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def test_declared_session_keeps_completion_responses_and_collection():
    session = Session([StreamEvent('vision', 'x', 0.0), StreamEvent('audio', 'y', 0.0)],
                      ['neural:IT'], time_bins=[(0, 10)])
    assert session.declare_inputs({'vision'}) is session
    assert session.complete is True
    assert session.time_bins == [(0, 10)]
    assert session.requested_output_channels == ('neural:IT',)
    assert session.next_input().channel == 'vision'
    assert session.next_input() is None
    session.respond(StreamEvent('neural:IT', np.array([1.0, 2.0]), 0.0))
    assert session.emitted[0].payload.tolist() == [1.0, 2.0]
    assert session.collect('neural:IT').values.tolist() == [[1.0, 2.0]]


def test_declared_session_carries_measurement_settings():
    session = Session([StreamEvent('vision', 'x', 0.0)], ['neural:IT'],
                      settings={'number_of_trials': 3})
    assert session.declare_inputs({'vision'}).settings == {'number_of_trials': 3}


def test_filter_drops_undeclared_channels_by_exact_match():
    events = [StreamEvent(channel, 'x', 0.0)
              for channel in ('vision', 'vision:image', 'text', 'audio')]
    session = Session(events).declare_inputs({'vision:image', 'text'})
    seen = []
    while (event := session.next_input()) is not None:
        seen.append(event.channel)
    assert seen == ['vision:image', 'text']


def test_control_events_pass_only_when_declared():
    events = [begin_event(0.0), StreamEvent('audio', 'y', 0.0), respond_event(1.0),
              end_event('terminal', 2.0)]
    session = Session(events).declare_inputs({'text', 'control'})
    delivered = []
    while (event := session.next_input()) is not None:
        delivered.append(event)
    assert [e.channel for e in delivered] == [CONTROL_CHANNEL] * 3
    assert [e.payload for e in delivered] == ['begin', 'respond', 'end']
    assert delivered[-1].meta == {'reason': 'terminal'}


def test_control_events_are_not_delivered_unless_asked_for():
    events = [begin_event(0.0), StreamEvent('text', 'a', 0.0), respond_event(1.0),
              end_event('terminal', 2.0)]
    for session in (Session(events), Session(events).declare_inputs({'text'})):
        assert [e.channel for e in iter(session.next_input, None)] == ['text']


def test_a_second_declaration_narrows_the_first():
    session = Session([StreamEvent('text', 'a', 0.0), StreamEvent('vision', 'b', 0.0)])
    session.declare_inputs({'text', 'vision'}).declare_inputs({'text'})
    assert session.next_input().channel == 'text'
    assert session.next_input() is None


def test_table_loses_undeclared_columns_and_keeps_identity_otherwise():
    table = StimulusSet(pd.DataFrame({'stimulus_id': ['a'], 'image_path': ['/x.png'],
                                      'sentence': ['hi']}))
    table.identifier = 'two-streams'
    kept = Session((), stimulus_set=table).declare_inputs({'vision', 'text'})
    assert kept.stimulus_set is table
    projected = Session((), stimulus_set=table).declare_inputs({'vision'})
    assert sorted(projected.stimulus_set.columns) == ['image_path', 'stimulus_id']
    assert projected.stimulus_set.identifier == 'two-streams'
    assert 'sentence' in table.columns        # the caller's table is never changed
