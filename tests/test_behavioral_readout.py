"""Tests for BrainScoreModel behavioral readout.

Validates the ProbabilitiesClassifier pattern ported into brainscore_core:
- start_task(TaskContext(task_type='probabilities', fitting_stimuli=...)) fits a classifier
- subsequent process() returns BehavioralAssembly with per-label probabilities
- labels and choice coordinate are preserved
- reset() clears the classifier
- missing behavioral_readout_layer raises a clear error
"""

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from brainscore_core.behavior import ProbabilitiesClassifier
from brainscore_core.model_interface import BrainScoreModel, TaskContext
from brainscore_core.supported_data_standards.brainio.assemblies import (
    BehavioralAssembly, NeuroidAssembly,
)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet


# ── ProbabilitiesClassifier unit tests ─────────────────────────

def _make_features(n_samples: int, n_features: int = 8, seed: int = 0) -> NeuroidAssembly:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_samples, n_features)).astype(np.float64)
    return NeuroidAssembly(
        data,
        coords={
            'stimulus_id': ('presentation', [f's{i}' for i in range(n_samples)]),
            'neuroid_id': ('neuroid', [f'n{j}' for j in range(n_features)]),
        },
        dims=['presentation', 'neuroid'],
    )


class TestProbabilitiesClassifier:
    def test_fit_and_predict_shape(self):
        X = _make_features(12)
        y = ['cat'] * 6 + ['dog'] * 6
        clf = ProbabilitiesClassifier()
        clf.fit(X, y)
        pred = clf.predict_proba(X)
        assert isinstance(pred, BehavioralAssembly)
        assert pred.dims == ('presentation', 'choice')
        assert pred.shape == (12, 2)

    def test_probabilities_sum_to_one(self):
        X = _make_features(20)
        y = ['a', 'b', 'c'] * 6 + ['a', 'b']
        clf = ProbabilitiesClassifier()
        clf.fit(X, y)
        pred = clf.predict_proba(X)
        row_sums = pred.values.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_choice_coord_preserves_label_order(self):
        X = _make_features(9)
        y = ['z', 'a', 'z', 'm', 'a', 'z', 'm', 'a', 'z']  # first seen order: z, a, m
        clf = ProbabilitiesClassifier()
        clf.fit(X, y)
        assert clf.label_set == ['z', 'a', 'm']
        pred = clf.predict_proba(X)
        assert list(pred.coords['choice'].values) == ['z', 'a', 'm']

    def test_predict_before_fit_raises(self):
        X = _make_features(4)
        clf = ProbabilitiesClassifier()
        with pytest.raises(RuntimeError, match="before fit"):
            clf.predict_proba(X)

    def test_predict_needs_2d(self):
        clf = ProbabilitiesClassifier()
        clf.fit(_make_features(10), ['a'] * 5 + ['b'] * 5)
        bad = np.ones((3, 4, 5))
        with pytest.raises(ValueError, match="2-D"):
            clf.predict_proba(bad)


# ── BrainScoreModel behavioral integration tests ───────────────

def _fake_preprocessor_returning(features_matrix):
    """Return a callable that behaves like a TextWrapper/VLMVisionWrapper
    — has an `identifier` so BrainScoreModel duck-types it as a full
    extractor that takes (stimuli, layers) and returns a NeuroidAssembly.
    """
    class FakeExtractor:
        identifier = 'fake-extractor'

        def __call__(self, stimuli, layers=None, **kwargs):
            n = len(stimuli)
            # return features for the requested number of stimuli
            data = features_matrix[:n] if features_matrix.shape[0] >= n else \
                np.resize(features_matrix, (n, features_matrix.shape[1]))
            return NeuroidAssembly(
                data,
                coords={
                    'stimulus_id': ('presentation', list(stimuli['stimulus_id'].values)),
                    'neuroid_id': ('neuroid', [f'n{j}' for j in range(data.shape[1])]),
                },
                dims=['presentation', 'neuroid'],
            )
    return FakeExtractor()


def _make_image_stimulus_set(labels, identifier='test_stim'):
    n = len(labels)
    stimuli = StimulusSet(pd.DataFrame({
        'stimulus_id': [f's{i}' for i in range(n)],
        'image_file_name': [f'/tmp/s{i}.png' for i in range(n)],
        'image_label': labels,
    }))
    stimuli.identifier = identifier
    return stimuli


class TestBrainScoreModelBehavioral:
    @pytest.fixture
    def labeled_stimuli(self):
        return _make_image_stimulus_set(['cat'] * 6 + ['dog'] * 6)

    @pytest.fixture
    def test_stimuli(self):
        return _make_image_stimulus_set(['cat'] * 3 + ['dog'] * 3, identifier='test_stim2')

    def test_start_task_fits_classifier(self, labeled_stimuli):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(12, 8))
        extractor = _fake_preprocessor_returning(features)
        model = BrainScoreModel(
            identifier='test-model',
            model=None,
            region_layer_map={},
            preprocessors={'vision': extractor},
            behavioral_readout_layer='some_layer',
        )
        assert model._readout_classifier is None

        model.start_task(TaskContext(
            task_type='probabilities',
            fitting_stimuli=labeled_stimuli,
            label_set=['cat', 'dog'],
        ))
        assert model._readout_classifier is not None
        assert model._readout_classifier.fitted

    def test_process_returns_behavioral_assembly(self, labeled_stimuli, test_stimuli):
        features = np.random.default_rng(42).normal(size=(12, 8))
        # Make labels linearly separable for stability
        features[:6, 0] += 3.0  # cat rows shift
        features[6:, 0] -= 3.0  # dog rows shift
        extractor = _fake_preprocessor_returning(features)
        model = BrainScoreModel(
            identifier='test-model',
            model=None,
            region_layer_map={},
            preprocessors={'vision': extractor},
            behavioral_readout_layer='some_layer',
        )
        model.start_task(TaskContext(
            task_type='probabilities',
            fitting_stimuli=labeled_stimuli,
        ))
        result = model.process(test_stimuli)
        assert isinstance(result, BehavioralAssembly)
        assert result.dims == ('presentation', 'choice')
        assert result.shape == (6, 2)
        # Sums to 1
        assert np.allclose(result.values.sum(axis=1), 1.0, atol=1e-6)

    def test_process_returns_neural_before_task(self, labeled_stimuli):
        features = np.random.default_rng(0).normal(size=(12, 8))
        extractor = _fake_preprocessor_returning(features)
        model = BrainScoreModel(
            identifier='test-model',
            model=None,
            region_layer_map={},
            preprocessors={'vision': extractor},
            behavioral_readout_layer='some_layer',
        )
        # Without start_task, process() returns neural features
        result = model.process(labeled_stimuli)
        assert isinstance(result, NeuroidAssembly)
        assert 'neuroid' in result.dims

    def test_reset_clears_classifier(self, labeled_stimuli):
        features = np.random.default_rng(0).normal(size=(12, 8))
        extractor = _fake_preprocessor_returning(features)
        model = BrainScoreModel(
            identifier='test-model',
            model=None,
            region_layer_map={},
            preprocessors={'vision': extractor},
            behavioral_readout_layer='some_layer',
        )
        model.start_task(TaskContext(
            task_type='probabilities',
            fitting_stimuli=labeled_stimuli,
        ))
        assert model._readout_classifier is not None
        model.reset()
        assert model._readout_classifier is None

    def test_missing_readout_layer_raises(self, labeled_stimuli):
        features = np.random.default_rng(0).normal(size=(12, 8))
        extractor = _fake_preprocessor_returning(features)
        model = BrainScoreModel(
            identifier='test-model',
            model=None,
            region_layer_map={},
            preprocessors={'vision': extractor},
            # No behavioral_readout_layer
        )
        with pytest.raises(ValueError, match="behavioral_readout_layer"):
            model.start_task(TaskContext(
                task_type='probabilities',
                fitting_stimuli=labeled_stimuli,
            ))

    def test_passive_task_does_not_fit(self, labeled_stimuli):
        features = np.random.default_rng(0).normal(size=(12, 8))
        extractor = _fake_preprocessor_returning(features)
        model = BrainScoreModel(
            identifier='test-model',
            model=None,
            region_layer_map={},
            preprocessors={'vision': extractor},
            behavioral_readout_layer='some_layer',
        )
        # 'passive' is neural-style, no readout needed
        model.start_task(TaskContext(task_type='passive'))
        assert model._readout_classifier is None
