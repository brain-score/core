"""Contract v3 rule 5.5: a stream the subject cannot take in never enters it."""
import numpy as np
import pandas as pd
import pytest

from brainscore_core.events import EnvironmentResponse, EnvironmentStep
from brainscore_core.model_interface import BrainScoreModel
from brainscore_core.session_view import DeclaredInputSession
from brainscore_core.streaming import BufferedSession, StreamEvent
from brainscore_core.streaming_helpers import neural_response
from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


class _VisionExtractor:
    def __init__(self):
        self.seen = []

    def __call__(self, stimuli, layers=None, **kwargs):
        self.seen.append(stimuli)
        n = len(stimuli)
        return NeuroidAssembly(
            np.ones((n, 2)), dims=['presentation', 'neuroid'],
            coords={'stimulus_id': ('presentation', list(stimuli['stimulus_id'])),
                    'neuroid_id': ('neuroid', [0, 1]),
                    'layer': ('neuroid', [layers[0]] * 2)})


def _model():
    extractor = _VisionExtractor()
    model = BrainScoreModel('vision-only', region_layer_map={'IT': 'layer'},
                            extractor={'vision': extractor})
    return model, extractor


def _movie():
    stimuli = StimulusSet(pd.DataFrame({
        'stimulus_id': ['clip0', 'clip1'],
        'image_path': ['/tmp/0.png', '/tmp/1.png'],
        'sentence': ['subtitle zero', 'subtitle one'],
        'audio_path': ['/tmp/0.wav', '/tmp/1.wav'],
        'object_name': ['cat', 'dog'],
    }))
    stimuli.identifier = 'movie-clips'
    return stimuli


def test_undeclared_modalities_never_reach_the_extractor():
    model, extractor = _model()
    response = neural_response(model, _movie(), record='IT')
    assert response.sizes['presentation'] == 2
    seen, = extractor.seen
    assert sorted(seen.columns) == ['image_path', 'object_name', 'stimulus_id']
    assert seen.identifier == 'movie-clips'  # cache identity survives the projection


def test_stimulus_set_keeps_its_identity_when_nothing_is_dropped():
    model, extractor = _model()
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': ['a'], 'image_path': ['/tmp/a.png']}))
    stimuli.identifier = 'one-image'
    neural_response(model, stimuli, record='IT')
    assert extractor.seen[0] is stimuli


def test_undeclared_events_are_skipped_and_declared_ones_delivered():
    model, extractor = _model()
    session = BufferedSession(inputs=[
        StreamEvent('text', 'not for this subject', 0.0,
                    meta={'stimulus_id': 't', 'column': 'sentence'}),
        StreamEvent('vision', '/tmp/a.png', 0.0,
                    meta={'stimulus_id': 'a', 'column': 'image_path'}),
    ], requested_output_channels=['neural:IT'])
    model.interact(session)
    seen, = extractor.seen
    assert list(seen['stimulus_id']) == ['a']
    assert session.collect('neural:IT').sizes['presentation'] == 1


def test_environment_steps_pass_through_to_the_policy():
    steps = []
    model = BrainScoreModel('policy', policy=lambda step: steps.append(step) or EnvironmentResponse(action=1))
    step = EnvironmentStep(observation={'value': 1})
    session = BufferedSession([step], requested_output_channels=['motor'])
    model.interact(session)
    assert steps == [step]
    assert session.emitted[0].payload.action == 1


def test_view_forwards_completion_responses_and_collection():
    inner = BufferedSession([StreamEvent('vision', 'x', 0.0), StreamEvent('audio', 'y', 0.0)],
                            requested_output_channels=['neural:IT'], time_bins=[(0, 10)])
    view = DeclaredInputSession(inner, {'vision'})
    assert view.complete is inner.complete
    assert view.time_bins == [(0, 10)]
    assert view.requested_output_channels == ('neural:IT',)
    assert view.next_input().channel == 'vision'
    assert view.next_input() is None
    view.respond(StreamEvent('neural:IT', 'out', 0.0))
    assert inner.emitted[0].payload == 'out'
    assert view.collect('neural:IT') == inner.collect('neural:IT')


def test_view_carries_measurement_settings():
    inner = BufferedSession([StreamEvent('vision', 'x', 0.0)], requested_output_channels=['neural:IT'])
    inner.settings = {'number_of_trials': 3}
    assert DeclaredInputSession(inner, {'vision'}).settings == {'number_of_trials': 3}
