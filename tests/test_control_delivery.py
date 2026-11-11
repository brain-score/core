"""The control signals reach only a subject that asks for them."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from brainscore_core.contract import Subject
from brainscore_core._evaluation.compatibility import CompatibilityError, check_channel_compatibility
from brainscore_core.streaming import (
    Session, StreamEvent, begin_event, end_event, respond_event,
)
from brainscore_core.streaming_helpers import EnvironmentSource, table_session
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def _table(n=4):
    table = StimulusSet(pd.DataFrame({'stimulus_id': [f's{i}' for i in range(n)],
                                      'sentence': [f'word {i}' for i in range(n)]}))
    table.identifier = 'control-delivery'
    return table


class _Reader(Subject):
    """The simplest subject: answers every event it is given."""
    identifier = 'reader'
    in_channels = {'text'}
    out_channels = {'behavior'}

    def __init__(self, control=False, answers=1):
        self.control, self.answers, self.seen = control, answers, []

    def participate(self, session):
        session.declare_inputs(self.in_channels | ({'control'} if self.control else set()))
        while (event := session.next_input()) is not None:
            self.seen.append(event.channel)
            for _ in range(self.answers if event.channel != 'control' else 0):
                session.respond(StreamEvent('behavior', 'yes', event.t_ms))


def test_a_subject_that_answers_every_event_gets_one_event_per_row():
    subject = _Reader()
    session = table_session(_table(), ['behavior'])
    subject.participate(session)
    assert subject.seen == ['text'] * 4 and len(session.emitted) == 4


def test_a_subject_that_declares_control_receives_the_signals():
    subject = _Reader(control=True)
    session = table_session(_table(2), ['behavior'])
    subject.participate(session)
    assert subject.seen == ['control', 'text', 'control'] * 2


def test_a_session_nobody_declared_for_delivers_no_control_events():
    events = [begin_event(0.0), StreamEvent('text', 'a', 0.0), respond_event(1.0), end_event('terminal', 2.0)]
    assert [e.channel for e in iter(Session(events).next_input, None)] == ['text']


def test_answering_a_trial_twice_names_the_channel_and_the_trial():
    session = table_session(_table(), ['behavior'])
    with pytest.raises(ValueError, match=r"'behavior'.*trial 1"):
        _Reader(answers=2).participate(session)


def test_several_channels_per_trial_and_a_late_batch_answer_are_fine():
    class Both(_Reader):
        out_channels = {'behavior', 'neural:IT'}

        def participate(self, session):
            session.declare_inputs(self.in_channels)
            events = list(iter(session.next_input, None))      # read every row, then answer
            for event in events:
                session.respond(StreamEvent('behavior', 'yes', event.t_ms))
                session.respond(StreamEvent('neural:IT', np.zeros(2), event.t_ms))
    session = table_session(_table(), ['behavior', 'neural:IT'])
    Both().participate(session)
    assert len(session.emitted) == 8


class _Remembering(_Reader):
    @property
    def traits(self):
        return {'memory:trial_history'}


def test_a_memory_subject_without_control_is_not_applicable():
    bench = SimpleNamespace(identifier='b', required_input_channels={'text'},
                            requested_output_channels={'behavior'})
    with pytest.raises(CompatibilityError, match="'control'"):
        check_channel_compatibility(_Remembering(), bench)
    _Remembering.in_channels = {'text', 'control'}
    try:
        match = check_channel_compatibility(_Remembering(), bench)
    finally:
        del _Remembering.in_channels
    assert match.consumed_inputs == {'text'} and not match.dropped_inputs


def test_control_is_not_a_benchmark_input():
    bench = SimpleNamespace(identifier='b', required_input_channels={'text', 'control'},
                            requested_output_channels={'behavior'})
    match = check_channel_compatibility(SimpleNamespace(
        identifier='s', in_channels={'text'}, out_channels={'behavior'},
        required_channels=set()), bench)
    assert match.consumed_inputs == {'text'} and not match.dropped_inputs


def test_a_closed_loop_subject_without_control_gets_a_clear_error():
    class Env:
        def reset(self):
            return {'observation': 1, 'reward': 0.0}

        def step(self, action):
            return {'observation': 2, 'reward': 0.0}, 'terminal'

    class Eager(_Reader):
        in_channels = {'observation', 'reward'}
        out_channels = {'motor'}

        def participate(self, session):
            session.declare_inputs(self.in_channels)
            while (event := session.next_input()) is not None:
                session.respond(StreamEvent('motor', 0, event.t_ms))
    session = Session(EnvironmentSource(Env()), ('motor',))
    with pytest.raises(RuntimeError, match="declare the 'control' input"):
        Eager().participate(session)
