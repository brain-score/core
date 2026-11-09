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


def test_an_edited_task_in_a_later_session_fits_again():
    """Review finding P1: a remembered readout is reused only for the full, unchanged task."""
    model, extractor = _model()
    task = _task()
    behavioral_response(model, task)
    assert _fits(extractor) == 1
    task.readout_fit = 'intact'                     # a setting the old fingerprint left out
    behavioral_response(model, task)
    assert _fits(extractor) == 2
    task.fitting_stimuli['image_path'] = [f'/tmp/other{i}.png' for i in range(4)]  # edited in place
    behavioral_response(model, task)
    assert _fits(extractor) == 3
    task.fitting_stimuli.stimulus_paths = {'f0': '/tmp/elsewhere.png'}
    behavioral_response(model, task)
    assert _fits(extractor) == 4
    behavioral_response(model, task)                 # unchanged since: reused
    assert _fits(extractor) == 4


def test_a_failed_task_setup_never_leaves_a_half_built_readout():
    """Review finding P2: a fit that fails part way is not remembered or reused."""
    model, _ = _model()
    a = _task(labels=('bird', 'fish'))
    model.start_task(a)
    one_class = _task(labels=('cat', 'cat'))
    with pytest.raises(ValueError):
        behavioral_response(model, one_class)        # the classifier cannot fit one class
    assert model._behavioral.task_context is a
    with pytest.raises(ValueError):
        model.start_task(one_class)                  # the same failure, outside a session
    model.start_task(_task(labels=('cat', 'dog')))   # another task, configured directly
    model.start_task(a)
    assert sorted(model.process(_images())['choice'].values) == ['bird', 'fish']


def test_a_neutral_session_keeps_the_remembered_readout():
    """Review finding P2: a session that configures no task does not cost a refit."""
    model, extractor = _model()
    model.start_task(_task(labels=('bird', 'fish')))  # the caller's task A
    model.start_recording('IT')
    b = _task()
    behavioral_response(model, b)
    assert _fits(extractor) == 2
    neural_response(model, _images(), record='IT')
    behavioral_response(model, b)
    assert _fits(extractor) == 2
    assert sorted(model.process(_images())['choice'].values) == ['bird', 'fish']


def test_a_reset_inside_a_session_stands(monkeypatch):
    """Review finding P2: the scope does not bring back a task that reset() cleared."""
    from brainscore_core.capabilities.sessions import NeuralSessions
    from brainscore_core.streaming import BufferedSession
    model, _ = _model()
    model.start_task(_task(labels=('bird', 'fish')))
    monkeypatch.setattr(NeuralSessions, 'participate',
                        lambda self, subject, session: subject.reset())
    model.participate(BufferedSession(stimulus_set=_images(),
                                      requested_output_channels=['neural:IT']))
    assert model._behavioral.task_context is None
    assert model._behavioral.readout_classifier is None
    model.start_recording('IT')
    assert not isinstance(model.process(_images()), BehavioralAssembly)


def test_an_unhashable_fitting_table_fits_no_more_often_than_before():
    """Review round 2, P1: a list-valued column falls back to the id-and-label check (rule 10.1)."""
    from brainscore_core.legacy_facade import LegacyBenchmarkFacade
    model, extractor = _model()
    fitting = _labelled()
    fitting['boxes'] = [[0, 1], [2, 3], [4, 5], [6, 7]]    # pandas cannot hash lists
    facade = LegacyBenchmarkFacade(model, visual_degrees=8)
    facade.start_task('probabilities', fitting_stimuli=fitting)
    facade.look_at(_images())
    facade.look_at(_images())
    assert _fits(extractor) == 1                         # the base count
    task = TaskContext(task_type='probabilities', fitting_stimuli=fitting,
                       metadata={'stimulus_set': _images()})
    behavioral_response(model, task)
    behavioral_response(model, task)
    assert _fits(extractor) == 2                         # a new task object fits once
