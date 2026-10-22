"""Public lab vocabulary preserves existing plugin and session behavior."""
import numpy as np
import pytest

from brainscore_core.model_interface import (
    BrainScoreModel, EnvironmentStep, EnvironmentResponse, StateChange,
    PerturbationApplied, Perturbation, Selection,
)
from brainscore_core.streaming import BufferedSession, InMemorySession, Session, StreamEvent


def test_buffered_session_retains_existing_import_and_delivery():
    assert InMemorySession is BufferedSession
    event = StreamEvent(channel='text', payload='hello', t_ms=0)
    session = BufferedSession(inputs=[event])
    assert session.next_input() is event
    assert session.next_input() is None
    session.respond(event)
    session.emit(event)
    assert session.collect() == [event, event]


def test_new_source_accepts_old_subject_and_old_source_accepts_new_subject():
    class NewSource(Session):
        def next_input(self):
            return None

        def respond(self, event):
            self.response = event

    class OldSource(Session):
        def next_input(self):
            return None

        def emit(self, event):
            self.response = event

    event = StreamEvent(channel='behavior', payload='yes', t_ms=None)
    new, old = NewSource(), OldSource()
    new.emit(event)
    old.respond(event)
    assert new.response is old.response is event


@pytest.mark.parametrize('public,legacy', [
    ('extractor', 'activations_model'), ('answer', 'generation_fn'),
    ('policy', 'action_fn'), ('intervention', 'state_change_fn'),
])
def test_conflicting_component_spellings_fail_before_execution(public, legacy):
    with pytest.raises(TypeError, match='not both'):
        BrainScoreModel(identifier='conflict', **{public: lambda: None, legacy: lambda: None})


@pytest.mark.parametrize('public,legacy', [
    ('answer', 'generation_fn'), ('policy', 'action_fn'), ('intervention', 'state_change_fn'),
])
def test_legacy_config_cannot_silently_override_public_component(public, legacy):
    with pytest.raises(TypeError, match='multiple'):
        BrainScoreModel(identifier='conflict', capability_config={legacy: lambda: None},
                        **{public: lambda: None})


def test_policy_dispatch_is_identical_with_both_spellings():
    response = EnvironmentResponse(action=np.arange(7, dtype=float))
    policy = lambda event: response
    event = EnvironmentStep()
    new = BrainScoreModel(identifier='policy', policy=policy)
    old = BrainScoreModel(identifier='policy', action_fn=policy)
    assert new.process(event) is old.process(event) is response


def test_intervention_cleanup_on_reset():
    active = []
    def intervene(change):
        active.append('installed')
        return PerturbationApplied(handle_id='one', target=change.target,
                                   perturbation=change.perturbation), active.clear

    subject = BrainScoreModel(identifier='intervention', intervention=intervene)
    subject.process(StateChange(kind='ablation', target=Selection(layer='layer'),
                                perturbation=Perturbation(kind='zero')))
    assert active == ['installed']
    subject.reset()
    assert active == []


@pytest.mark.parametrize('modality,column', [('vision', 'image_path'), ('text', 'sentence'),
                                          ('audio', 'audio_path'), ('video', 'video_path')])
def test_extractor_and_buffered_session_work_across_modalities(modality, column):
    import xarray as xr
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

    def extract(stimuli, *, layers):
        return xr.DataArray(np.ones((len(stimuli), 2)), dims=('presentation', 'neuroid'),
                            coords={'layer': ('neuroid', [layers[0]] * 2),
                                    'neuroid_id': ('neuroid', [0, 1])})
    subject = BrainScoreModel(identifier='extractor', extractor={modality: extract},
                              region_layer_map={'region': 'layer'})
    session = BufferedSession(stimulus_set=StimulusSet({column: ['one', 'two']}),
                              requested_output_channels=('neural:region',))
    subject.interact(session)
    np.testing.assert_array_equal(session.collect().values, np.ones((2, 2)))


def test_answer_matches_legacy_behavioral_output():
    from brainscore_core.model_interface import TaskContext
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    import xarray as xr

    def answer(stimulus_row, instruction, label_set):
        return label_set[0] if stimulus_row.sentence == 'first' else label_set[1]
    stimuli = StimulusSet({'sentence': ['first', 'second'], 'stimulus_id': ['a', 'b']})
    task = TaskContext(task_type='label', instruction='choose', label_set=['yes', 'no'])
    outputs = []
    for spelling in ('answer', 'generation_fn'):
        subject = BrainScoreModel(identifier='answer', supported_modalities={'text'},
                                  **{spelling: answer})
        subject.start_task(task)
        outputs.append(subject.process(stimuli))
    xr.testing.assert_identical(*outputs)


@pytest.mark.parametrize('spelling', ['emit', 'respond'])
def test_source_subclasses_can_extend_responses_with_super(spelling):
    class LegacyExtension(BufferedSession):
        def emit(self, event):
            self.calls = getattr(self, 'calls', 0) + 1
            super().emit(event)

    class NewExtension(LegacyExtension):
        def respond(self, event):
            self.new_calls = getattr(self, 'new_calls', 0) + 1
            super().respond(event)

    event = StreamEvent(channel='behavior', payload='answer', t_ms=0)
    session = NewExtension()
    getattr(session, spelling)(event)
    assert session.collect() == [event]
    assert session.calls == session.new_calls == 1
