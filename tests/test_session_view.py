"""Contract v3 rule 5.5: a stream the subject cannot take in never enters it."""

import numpy as np
import pandas as pd

from brainscore_core.streaming import (
    CONTROL_CHANNEL, Session, StreamEvent, begin_event, end_event, respond_event,
)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


def test_declared_session_keeps_completion_responses_and_collection():
    session = Session([StreamEvent('vision:image', 'x', 0.0), StreamEvent('audio', 'y', 0.0)],
                      ['neural:IT'], time_bins=[(0, 10)])
    assert session.declare_inputs({'vision:image'}) is session
    assert session.complete is True
    assert session.time_bins == [(0, 10)]
    assert session.outputs == ('neural:IT',)
    assert session.next_input().channel == 'vision:image'
    assert session.next_input() is None
    session.respond(StreamEvent('neural:IT', np.array([1.0, 2.0]), 0.0))
    assert session.emitted[0].payload.tolist() == [1.0, 2.0]
    assert session.collect('neural:IT').values.tolist() == [[1.0, 2.0]]


def test_declared_session_carries_measurement_settings():
    session = Session([StreamEvent('vision:image', 'x', 0.0)], ['neural:IT'],
                      settings={'number_of_trials': 3})
    assert session.declare_inputs({'vision:image'}).settings == {'number_of_trials': 3}


def test_filter_drops_undeclared_channels_by_exact_match():
    events = [StreamEvent(channel, 'x', 0.0)
              for channel in ('vision:video', 'vision:image', 'text', 'audio')]
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
    session = Session([StreamEvent('text', 'a', 0.0), StreamEvent('vision:image', 'b', 0.0)])
    session.declare_inputs({'text', 'vision:image'}).declare_inputs({'text'})
    assert session.next_input().channel == 'text'
    assert session.next_input() is None


def test_table_loses_undeclared_columns_and_keeps_identity_otherwise():
    table = StimulusSet(pd.DataFrame({'stimulus_id': ['a'], 'image_path': ['/x.png'],
                                      'sentence': ['hi']}))
    table.identifier = 'two-streams'
    kept = Session((), stimulus_set=table).declare_inputs({'vision:image', 'text'})
    assert kept.stimulus_set is table
    projected = Session((), stimulus_set=table).declare_inputs({'vision:image'})
    assert sorted(projected.stimulus_set.columns) == ['image_path', 'stimulus_id']
    assert projected.stimulus_set.identifier == 'two-streams'
    assert 'sentence' in table.columns        # the caller's table is never changed


def _trial_table():
    return StimulusSet(pd.DataFrame({'stimulus_id': ['a'], 'sentence': ['hi'],
                                     'image_path': ['/x.png'], 'video_path': ['/x.mp4']}))


def test_task_loses_the_parts_of_undeclared_inputs_and_the_callers_task_is_unchanged():
    from brainscore_core.contract import TaskContext
    from brainscore_core.responses import Text, trial_prompt
    task = TaskContext(response=Text(), instruction='Rate it.',
                       present=('text', 'image:image_path', 'video'))
    session = Session((), ['behavior:text'], task, stimulus_set=_trial_table())
    session.declare_inputs({'text', 'instruction'})
    assert session.task.present == ('text',)
    assert task.present == ('text', 'image:image_path', 'video')
    assert (session.task.instruction, session.task.response) == (task.instruction, task.response)
    row = session.stimulus_set.iloc[0]
    assert trial_prompt(session.task, row).names == ('instruction', 'text')


def test_task_of_a_full_match_is_the_same_object():
    from brainscore_core.contract import TaskContext
    from brainscore_core.responses import Text
    task = TaskContext(response=Text(), present=('text', 'image:image_path', 'video'))
    session = Session((), ['behavior:text'], task, stimulus_set=_trial_table())
    session.declare_inputs({'text', 'vision:image', 'vision:video'})
    assert session.task is task


def test_a_present_entry_no_input_provides_is_still_an_error():
    import pytest
    from brainscore_core.contract import TaskContext
    from brainscore_core.responses import Text, trial_prompt
    # Dropped for this subject, but the column exists nowhere: refused, not skipped.
    task = TaskContext(response=Text(), present=('text', 'image:image_pth'))
    session = Session((), ['behavior:text'], task, stimulus_set=_trial_table())
    with pytest.raises(ValueError, match='image_pth'):
        session.declare_inputs({'text'})
    # Kept for this subject: the trial still refuses the missing column.
    task = TaskContext(response=Text(), present=('text:sentense',))
    session = Session((), ['behavior:text'], task, stimulus_set=_trial_table())
    session.declare_inputs({'text'})
    with pytest.raises(ValueError, match='sentense'):
        trial_prompt(session.task, session.stimulus_set.iloc[0])
