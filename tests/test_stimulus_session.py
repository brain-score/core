"""The one stimulus session builder: any requested outputs, a task, and a streaming mode."""

import numpy as np
import pandas as pd
import pytest

from brainscore_core.contract import TaskContext, required_traits_for
from brainscore_core.streaming import Session, StreamEvent, _with_trial_memory
from brainscore_core.streaming_helpers import (
    drive_behavior_session,
    drive_behavior_session_streaming,
    drive_neural_session,
    stimulus_session,
)
from brainscore_core.supported_data_standards.brainio.assemblies import (
    BehavioralAssembly,
    NeuroidAssembly,
)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

pytestmark = pytest.mark.unit


def _table(n=3):
    stimuli = StimulusSet(pd.DataFrame({
        'stimulus_id': [f's{i}' for i in range(n)],
        'sentence': [f'word {i}' for i in range(n)],
    }))
    stimuli.identifier = 'stimulus-session-test'
    return stimuli


def _neural(ids):
    return NeuroidAssembly(
        np.ones((len(ids), 2)),
        coords={'stimulus_id': ('presentation', list(ids)),
                'neuroid_id': ('neuroid', ['n0', 'n1']),
                'layer': ('neuroid', ['l', 'l'])},
        dims=['presentation', 'neuroid'])


def _behavior(ids):
    return BehavioralAssembly(
        np.full((len(ids), 2), 0.5),
        coords={'stimulus_id': ('presentation', list(ids)),
                'choice': ('choice', ['a', 'b'])},
        dims=['presentation', 'choice'])


class _Typed:
    def __init__(self):
        self.calls = []

    def start_recording(self, target, time_bins=None):
        self.calls.append(('start_recording', target, time_bins))

    def start_task(self, task_context):
        self.calls.append(('start_task', task_context))

    def process(self, stimuli, **kwargs):
        self.calls.append(('process', len(stimuli)))
        ids = list(stimuli['stimulus_id'])
        return _behavior(ids) if any(c[0] == 'start_task' for c in self.calls) else _neural(ids)


def _inputs(session):
    return [event for event in iter(session.next_input, None)]


def test_neural_outputs_keep_the_table_and_answer_in_one_call():
    stimuli = _table()
    session = stimulus_session(stimuli, ['neural:IT'], time_bins=[(70, 170)])
    assert session.requested_output_channels == ('neural:IT',)
    assert session.stimulus_set is stimuli and session.task_context is None
    assert session.streaming is False
    subject = _Typed()
    drive_neural_session(subject, session)
    assert subject.calls == [('start_recording', 'IT', [(70, 170)]), ('process', 3)]


def test_behavior_with_a_task_carries_the_task_and_presents_only_the_table():
    stimuli = _table()
    task = TaskContext(task_type='probabilities', label_set=['a', 'b'],
                       fitting_stimuli=_table(2), instruction='Pick one')
    session = stimulus_session(stimuli, ['behavior'], task)
    assert session.requested_output_channels == ('behavior',)
    assert session.task_context is task and session.stimulus_set is stimuli
    events = _inputs(session.declare_inputs({'text', 'instruction'}))
    # The instruction and the fitting rows stay on the task context.
    assert [event.payload for event in events] == ['word 0', 'word 1', 'word 2']
    subject = _Typed()
    drive_behavior_session(subject, stimulus_session(stimuli, ['behavior'], task))
    assert subject.calls == [('start_task', task), ('process', 3)]


def test_behavior_and_neural_together_mark_each_row_as_one_trial():
    task = TaskContext(task_type='probabilities', label_set=['a', 'b'])
    session = stimulus_session(_table(2), ['behavior', 'neural:IT'], task)
    assert session.requested_output_channels == ('behavior', 'neural:IT')
    assert session.one_answer_per_trial is True
    events = _inputs(session.declare_inputs({'text', 'control'}))
    assert [e.payload for e in events] == ['begin', 'word 0', 'end', 'begin', 'word 1', 'end']


@pytest.mark.parametrize('outputs', [['neural:IT'], ['behavior'], ['behavior', 'neural:IT'], None])
def test_streaming_makes_one_row_at_a_time_for_any_outputs(outputs):
    stimuli = _table(4)
    pulled = []
    rows = stimuli.iterrows

    def counting_rows():
        for row in rows():
            pulled.append(row[0])
            yield row

    stimuli.iterrows = counting_rows
    task = TaskContext(task_type='probabilities', label_set=['a', 'b'])
    session = stimulus_session(stimuli, outputs, task, streaming=True)
    assert pulled == [], 'building the session read the table'
    assert session.streaming is True and session.task_context is task
    assert session.requested_output_channels == (None if outputs is None else tuple(outputs))
    first = session.next_input()
    assert first.payload == 'word 0' and len(pulled) == 1
    rest = _inputs(session)
    assert [event.payload for event in rest] == ['word 1', 'word 2', 'word 3']


def test_streaming_behavior_answers_each_trial_after_one_task_setup():
    task = TaskContext(task_type='probabilities', label_set=['a', 'b'])
    session = stimulus_session(_table(3), ['behavior'], task, streaming=True)
    subject = _Typed()
    drive_behavior_session_streaming(subject, session)
    assert subject.calls == [('start_task', task)] + [('process', 1)] * 3
    assert len(session.emitted) == 3


@pytest.mark.parametrize('removed', [{'record': 'IT'}, {'state_change': object()},
                                     {'scoring_stimuli': _table()}])
def test_session_rejects_the_removed_arguments(removed):
    with pytest.raises(TypeError):
        Session((), ['neural:IT'], **removed)


def test_stimulus_session_rejects_record():
    with pytest.raises(TypeError):
        stimulus_session(_table(), record='IT')


def test_trial_memory_conversation_is_rejected():
    with pytest.raises(ValueError, match="'none', 'history'"):
        TaskContext(task_type='probabilities', trial_memory='conversation')
    with pytest.raises(ValueError):
        required_traits_for(None, 'conversation')
    with pytest.raises(ValueError):
        list(_with_trial_memory([], 'conversation'))


def test_trial_memory_history_needs_the_trial_history_trait():
    task = TaskContext(task_type='probabilities', trial_memory='history')
    assert task.trial_memory == 'history'
    assert required_traits_for(None, 'history') == {'memory:trial_history'}
    assert required_traits_for(None, 'none') == set()
    session = stimulus_session(_table(2), ['behavior'], task)
    assert session.one_answer_per_trial is False
    events = _inputs(session.declare_inputs({'text', 'control'}))
    assert [e.payload for e in events] == ['begin', 'word 0', 'word 1', 'end']


def test_history_benchmark_is_not_compatible_without_the_trait():
    from brainscore_core._evaluation.compatibility import CompatibilityError, check_compatibility
    from brainscore_core.contract import Subject

    class _Subject(Subject):
        identifier = 'plain'
        in_channels = {'text', 'control'}
        out_channels = {'behavior'}

        def __init__(self, traits=()):
            self._traits = set(traits)

        @property
        def traits(self):
            return self._traits

        def participate(self, session):
            pass

    class _Benchmark:
        identifier = 'history-bench'
        required_input_channels = {'text'}
        requested_output_channels = {'behavior'}
        trial_memory = 'history'

    with pytest.raises(CompatibilityError, match='memory:trial_history'):
        check_compatibility(_Subject(), _Benchmark())
    check_compatibility(_Subject({'memory:trial_history'}), _Benchmark())


def test_a_perturbation_answer_without_meta_is_read_against_the_delivered_change():
    """The session reads the state change from the event it delivered, not from an argument."""
    from brainscore_core._evaluation.measures import record_measures
    from brainscore_core._sessions.buffered import _state_change_session
    from brainscore_core.events import PerturbationApplied, StateChange
    with record_measures() as record:
        session = _state_change_session(StateChange(kind='reset', handle_id='h'))
        session.next_input()
        session.respond(StreamEvent('perturbation', PerturbationApplied('h', None, None), 0.0))
    assert record.undone == {'h'} and record.applied == set()


def test_the_state_change_driver_reads_the_change_the_source_delivered():
    """A declared bare family drops an addressed change event; the driver still reads it."""
    from brainscore_core._sessions.buffered import _state_change_session
    from brainscore_core.events import Perturbation, PerturbationApplied, Selection, StateChange
    from brainscore_core.streaming_helpers import drive_state_change_session
    change = StateChange(kind='ablation', target=Selection(layer='blocks.1'),
                         perturbation=Perturbation(kind='zero'))
    session = _state_change_session(change).declare_inputs({'lesion'})
    seen = []

    class _Subject:
        def process(self, state_change):
            seen.append(state_change)
            return PerturbationApplied('h', None, None)

    drive_state_change_session(_Subject(), session)
    assert seen == [change]
    assert session.collect('perturbation').handle_id == 'h'
