"""The same declarations apply to every subject and benchmark."""
from types import SimpleNamespace

import pytest

from brainscore_core.contract import Subject
from brainscore_core.compatibility import CompatibilityError, check_compatibility


class Participant(Subject):
    identifier = 'participant'
    in_channels = {'vision'}
    out_channels = {'behavior'}
    required_channels = set()

    def interact(self, session):
        raise AssertionError('Pre-flight must not run a subject')


def benchmark(inputs=('vision',), outputs=('behavior',)):
    return SimpleNamespace(identifier='experiment', required_input_channels=set(inputs),
                           requested_output_channels=set(outputs))


def test_partial_match_records_inputs_without_running_subject():
    match = check_compatibility(Participant(), benchmark(('vision', 'audio')))
    assert match.kind == 'partial'
    assert match.consumed_inputs == {'vision'}
    assert match.dropped_inputs == {'audio'}
    score = SimpleNamespace(attrs={})
    match.stamp(score)
    assert score.attrs == dict(compatibility='partial', consumed_inputs=['vision'],
                              dropped_inputs=['audio'], modalities_used=['vision'])


def test_full_match():
    assert check_compatibility(Participant(), benchmark()).kind == 'full'


@pytest.mark.parametrize('inputs,outputs', [
    (('audio',), ('behavior',)),
    (('vision',), ('neural:IT',)),
    ((), ('behavior',)),
])
def test_incompatible_is_rejected_before_compute(inputs, outputs):
    with pytest.raises(CompatibilityError):
        check_compatibility(Participant(), benchmark(inputs, outputs))


def test_required_inputs_are_not_optional():
    subject = Participant()
    subject.required_channels = {'audio'}
    with pytest.raises(CompatibilityError, match='audio'):
        check_compatibility(subject, benchmark())


def test_catalog_replacement_requires_explicit_consent():
    from brainscore_core import io_catalog
    entry = io_catalog.get('vision')
    with pytest.raises(ValueError, match='already registered'):
        io_catalog.register(entry)
    io_catalog.register(entry, replace=True)
    assert io_catalog.get('vision') is entry


def test_unwired_neural_path_is_not_declared():
    from brainscore_core.brainscore_model import BrainScoreModel
    model = BrainScoreModel(identifier='empty', region_layer_map={'IT': 'layer'})
    assert 'neural:IT' not in model.out_channels
    with pytest.raises(ValueError, match='Undeclared'):
        model.check_session_support({'neural:IT'})


def test_session_routing_rejects_missing_motor_before_reading_inputs():
    from brainscore_core.brainscore_model import BrainScoreModel
    from brainscore_core.streaming import BufferedSession
    model = BrainScoreModel(identifier='empty')
    session = BufferedSession(requested_output_channels=['motor'])
    with pytest.raises(ValueError, match='motor'):
        model.interact(session)


def test_partial_subject_never_receives_the_dropped_stream():
    from brainscore_core.session_view import InputSession
    from brainscore_core.streaming import BufferedSession, StreamEvent
    original = BufferedSession(inputs=[
        StreamEvent('text', 'not for this subject', None),
        StreamEvent('vision', 'image.png', None),
    ])
    view = InputSession(original, {'vision'})
    assert view.next_input().payload == 'image.png'
    assert view.next_input() is None


def test_buffered_payload_validation_happens_before_first_input():
    from brainscore_core.streaming import BufferedSession, StreamEvent
    with pytest.raises(ValueError, match='text'):
        BufferedSession(inputs=[StreamEvent('text', 'valid', None),
                                StreamEvent('text', 42, None)])


def test_explicit_extractor_never_inspects_its_signature(monkeypatch):
    import inspect
    from brainscore_core.brainscore_model import BrainScoreModel
    from brainscore_core.streaming_helpers import neural_response
    from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    calls = []
    def extract(stimuli, *, layers):
        calls.append(layers)
        return NeuroidAssembly([[1.]], dims=['presentation', 'neuroid'])
    monkeypatch.setattr(inspect, 'signature', lambda *a, **k: pytest.fail('Inspected extractor'))
    subject = BrainScoreModel(
        identifier='explicit', extractor={'text': extract}, region_layer_map={'language': 'layer'},
    )
    result = neural_response(subject, StimulusSet({'sentence': ['hello']}), record='language')
    assert result.item() == 1 and calls == [['layer']]


def test_environment_replay_validates_all_saved_observations():
    from brainscore_core.events import EnvironmentStep
    from brainscore_core.legacy_sessions import EnvironmentReplaySession
    first = EnvironmentStep(observation={'value': 1})
    with pytest.raises(TypeError, match='EnvironmentStep'):
        EnvironmentReplaySession([first, object()])
    session = EnvironmentReplaySession([first])
    assert session.next_input() is first
    assert session.next_input() is None


def test_channel_envelope_runs_the_same_policy():
    from brainscore_core.brainscore_model import BrainScoreModel
    from brainscore_core.events import EnvironmentStep, EnvironmentResponse
    from brainscore_core.streaming import BufferedSession, StreamEvent
    seen = []
    def policy(step):
        seen.append(step)
        return EnvironmentResponse(action=2)
    subject = BrainScoreModel(identifier='policy', policy=policy)
    step = EnvironmentStep(observation={'value': 1})
    session = BufferedSession(
        [StreamEvent('observation', step, None)], requested_output_channels=['motor'],
    )
    subject.interact(session)
    assert seen == [step]
    assert session.emitted[0].payload.action == 2
    assert session.emitted[0].t_ms is None
