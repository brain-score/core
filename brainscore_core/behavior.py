"""
Behavioral readout for BrainScoreModel.

Ports the `ProbabilitiesClassifier` pattern from `brainscore_vision.model_helpers
.brain_transformation.behavior` into `brainscore_core` so behavioral readout is
available to any UnifiedModel without depending on domain-specific code.

Usage (internal to BrainScoreModel):
    classifier = ProbabilitiesClassifier()
    classifier.fit(fitting_features, fitting_labels)
    behavioral_assembly = classifier.predict_proba(features)

`fitting_features` and `features` must be 2-D (presentation, neuroid) DataAssembly.
`fitting_labels` is an iterable of string labels, one per presentation.
"""

from collections import OrderedDict
from typing import Iterable

import numpy as np

from brainscore_core.supported_data_standards.brainio.assemblies import (
    BehavioralAssembly, array_is_element, walk_coords,
)


class ProbabilitiesClassifier:
    """Logistic regression readout that produces label probabilities.

    Scales features with StandardScaler then fits a multinomial logistic
    regression. At predict time, returns a BehavioralAssembly with shape
    (presentation, choice) where choice is the label_set derived from
    fitting labels.
    """

    def __init__(self, classifier_c: float = 1e-3):
        # Lazy-imported so brainscore_core stays light
        import sklearn.linear_model
        import sklearn.preprocessing
        self._classifier = sklearn.linear_model.LogisticRegression(
            multi_class='multinomial', solver='newton-cg', C=classifier_c,
        )
        self._scaler = sklearn.preprocessing.StandardScaler()
        self._label_mapping: 'OrderedDict[int, str]' = OrderedDict()
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def label_set(self) -> list:
        return list(self._label_mapping.values())

    def fit(self, features, labels: Iterable) -> 'ProbabilitiesClassifier':
        """Fit the logistic classifier.

        Args:
            features: 2-D (presentation, neuroid) array or DataAssembly.
            labels: iterable of label strings, length == n_presentations.
        """
        X = np.asarray(features)
        self._scaler.fit(X)
        X = self._scaler.transform(X)
        y, self._label_mapping = self._labels_to_indices(list(labels))
        self._classifier.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, features) -> BehavioralAssembly:
        """Predict per-label probabilities.

        Args:
            features: 2-D (presentation, neuroid) DataAssembly.

        Returns:
            BehavioralAssembly with dims (presentation, choice).
            Presentation coordinates are copied from the input.
        """
        if not self._fitted:
            raise RuntimeError(
                "ProbabilitiesClassifier.predict_proba called before fit()."
            )
        arr = np.asarray(features)
        if arr.ndim != 2:
            raise ValueError(
                f"Expected 2-D (presentation, neuroid) features; "
                f"got shape {arr.shape}."
            )
        scaled = self._scaler.transform(arr)
        proba = self._classifier.predict_proba(scaled)

        presentation_dim = features.dims[0] if hasattr(features, 'dims') else 'presentation'
        presentation_coords = {}
        if hasattr(features, 'coords'):
            presentation_coords = {
                coord: (dims, values)
                for coord, dims, values in walk_coords(features)
                if array_is_element(dims, presentation_dim)
            }

        return BehavioralAssembly(
            proba,
            coords={
                **presentation_coords,
                'choice': list(self._label_mapping.values()),
            },
            dims=[presentation_dim, 'choice'],
        )

    @staticmethod
    def _labels_to_indices(labels) -> 'tuple[list[int], OrderedDict[int, str]]':
        label_to_index: 'OrderedDict[str, int]' = OrderedDict()
        indices = []
        for label in labels:
            if label not in label_to_index:
                label_to_index[label] = len(label_to_index)
            indices.append(label_to_index[label])
        index_to_label: 'OrderedDict[int, str]' = OrderedDict(
            (idx, lbl) for lbl, idx in label_to_index.items()
        )
        return indices, index_to_label
