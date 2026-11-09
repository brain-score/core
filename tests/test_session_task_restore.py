"""A session that configures a task leaves the subject as it found it (rules 4.5, 8.3)."""
import numpy as np
import pandas as pd
import pytest

from brainscore_core.model_interface import BrainScoreModel, TaskContext
from brainscore_core.streaming_helpers import behavioral_response, neural_response
from brainscore_core.supported_data_standards.brainio.assemblies import (
    BehavioralAssembly, NeuroidAssembly)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


class _Extractor:
    """Deterministic features; fails on the scored stimuli when asked to."""

    def __init__(self):
        self.calls = []
        self.fail_on_scoring = False

    def __call__(self, stimuli, layers=None, **kwargs):
        ids = list(stimuli['stimulus_id'])
        self.calls.append(ids)
        if self.fail_on_scoring and ids and str(ids[0]).startswith('s'):
            raise RuntimeError('extraction failed')
        n = len(stimuli)
        return NeuroidAssembly(
            np.arange(n * 3, dtype=float).reshape(n, 3) + np.arange(n)[:, None] ** 2,
            coords={'stimulus_id': ('presentation', ids),
                    'neuroid_id': ('neuroid', np.arange(3)),
                    'layer': ('neuroid', [layers[0]] * 3)},
            dims=['presentation', 'neuroid'])


def _images(n=2):
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': [f's{i}' for i in range(n)],
                                        'image_path': [f'/tmp/s{i}.png' for i in range(n)]}))
    stimuli.identifier = f'images-{n}'
    return stimuli


def _labelled(n=4, labels=('cat', 'dog')):
    stimuli = StimulusSet(pd.DataFrame({'stimulus_id': [f'f{i}' for i in range(n)],
                                        'image_path': [f'/tmp/f{i}.png' for i in range(n)],
                                        'image_label': list(labels) * (n // 2)}))
    stimuli.identifier = f'fitting-{"-".join(labels)}'
    return stimuli


def _model():
    extractor = _Extractor()
    model = BrainScoreModel('restore', region_layer_map={'IT': 'layer'},
                            extractor={'vision': extractor}, behavioral_readout_layer='layer')
    return model, extractor


def _task(labels=('cat', 'dog')):
    return TaskContext(task_type='probabilities', fitting_stimuli=_labelled(labels=labels),
                       metadata={'stimulus_set': _images()})


def _fits(extractor):
    return sum(1 for ids in extractor.calls if ids and str(ids[0]).startswith('f'))


def test_process_records_activity_again_after_a_behavior_session():
    model, _ = _model()
    model.start_recording('IT')
    assert isinstance(behavioral_response(model, _task()), BehavioralAssembly)
    after = model.process(_images())
    assert isinstance(after, NeuroidAssembly) and not isinstance(after, BehavioralAssembly)
    assert model._behavioral.task_context is None and model._behavioral.readout_classifier is None


def test_the_callers_task_survives_a_behavior_session():
    model, _ = _model()
    prior = _task(labels=('bird', 'fish'))
    model.start_task(prior)
    classifier = model._behavioral.readout_classifier
    behavioral_response(model, _task())
    assert model._behavioral.task_context is prior
    assert model._behavioral.readout_classifier is classifier
    output = model.process(_images())
    assert sorted(output['choice'].values) == ['bird', 'fish']


def test_the_callers_task_returns_when_the_session_fails():
    model, extractor = _model()
    prior = _task(labels=('bird', 'fish'))
    model.start_task(prior)
    classifier = model._behavioral.readout_classifier
    extractor.fail_on_scoring = True
    with pytest.raises(RuntimeError, match='extraction failed'):
        behavioral_response(model, _task())
    assert model._behavioral.task_context is prior
    assert model._behavioral.readout_classifier is classifier


def test_the_same_task_in_a_later_session_keeps_its_fitted_readout():
    model, extractor = _model()
    task = _task()
    first = behavioral_response(model, task)
    assert _fits(extractor) == 1
    model.start_recording('IT')
    neural_response(model, _images(), record='IT')   # a session in between
    second = behavioral_response(model, task)
    assert _fits(extractor) == 1
    np.testing.assert_array_equal(first.values, second.values)
    model.start_task(task)                           # the caller sends the same task
    assert _fits(extractor) == 1
    assert isinstance(model.process(_images()), BehavioralAssembly)


def test_a_new_task_in_a_later_session_fits_again():
    model, extractor = _model()
    behavioral_response(model, _task())
    behavioral_response(model, _task(labels=('bird', 'fish')))
    assert _fits(extractor) == 2
