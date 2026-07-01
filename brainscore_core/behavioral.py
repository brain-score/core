"""Behavioral readout and generation support for BrainScoreModel."""

from typing import Any, Optional

from .contract import TaskContext


class BehavioralReadout:
    """Owns behavioral task state for a BrainScoreModel instance."""

    BEHAVIORAL_TASK_TYPES = frozenset({
        'probabilities', 'classification', 'label',
    })

    def __init__(self, owner) -> None:
        self.owner = owner
        self.task_context: Optional[TaskContext] = None
        self.readout_classifier: Any = None
        self.use_generation_for_task = False

    @classmethod
    def requires_behavioral_readout(cls, task_type: str) -> bool:
        return task_type in cls.BEHAVIORAL_TASK_TYPES

    def start_task(self, task_context_or_task, fitting_stimuli=None,
                   **kwargs) -> None:
        del kwargs
        owner = self.owner
        if isinstance(task_context_or_task, TaskContext):
            task_context = task_context_or_task
        else:
            task_context = TaskContext(
                task_type=task_context_or_task,
                fitting_stimuli=fitting_stimuli,
            )
        self.task_context = task_context

        task_type = task_context.task_type
        if not owner._requires_behavioral_readout(task_type):
            self.use_generation_for_task = False
            self.readout_classifier = None
            return

        instruction = task_context.instruction
        label_set = task_context.label_set
        fitting = task_context.fitting_stimuli
        prefer = task_context.prefer_path or 'auto'
        if prefer not in ('auto', 'generation', 'readout'):
            raise ValueError(
                f"TaskContext.prefer_path must be one of "
                f"('auto', 'generation', 'readout'); got {prefer!r}."
            )

        gen_viable = (owner._generation_fn is not None
                      and instruction and label_set)
        readout_viable = fitting is not None

        if prefer == 'generation':
            if not gen_viable:
                raise ValueError(
                    f"Model '{owner.identifier}' cannot run task "
                    f"'{task_type}' via generation path "
                    f"(prefer_path='generation'): "
                    f"requires model.generation_fn AND TaskContext.instruction "
                    f"AND TaskContext.label_set to be set."
                )
            self.use_generation_for_task = True
            self.readout_classifier = None
            return

        if prefer == 'readout':
            if not readout_viable:
                raise ValueError(
                    f"Model '{owner.identifier}' cannot run task "
                    f"'{task_type}' via readout path "
                    f"(prefer_path='readout'): "
                    f"requires TaskContext.fitting_stimuli to be set."
                )
            self.use_generation_for_task = False
            owner._fit_behavioral_readout(fitting)
            return

        if gen_viable:
            self.use_generation_for_task = True
            self.readout_classifier = None
            return

        self.use_generation_for_task = False
        if readout_viable:
            owner._fit_behavioral_readout(fitting)
        else:
            raise ValueError(
                f"Model '{owner.identifier}' cannot run behavioral task "
                f"'{task_type}': TaskContext provides neither fitting_stimuli "
                f"(needed for readout path) nor (instruction + label_set "
                f"+ model.generation_fn) (needed for generation path)."
            )

    def reset(self) -> None:
        self.task_context = None
        self.readout_classifier = None
        self.use_generation_for_task = False

    def fit_behavioral_readout(self, fitting_stimuli) -> None:
        """Extract readout-layer features and fit a probabilities classifier."""
        from brainscore_core.behavior import ProbabilitiesClassifier

        owner = self.owner
        task_type = (
            self.task_context.task_type
            if self.task_context is not None else None
        )
        if owner._behavioral_readout_layer is None:
            raise ValueError(
                f"Model '{owner.identifier}' was asked to perform a behavioral "
                f"task (task_type={task_type!r}) but no "
                f"behavioral_readout_layer is set. Register the model with "
                f"BrainScoreModel(..., behavioral_readout_layer='<layer name>')."
            )

        labels = self.extract_labels(fitting_stimuli)
        features = owner._extract_behavioral_features(fitting_stimuli)

        self.readout_classifier = ProbabilitiesClassifier()
        self.readout_classifier.fit(features, labels)

    @staticmethod
    def extract_labels(stimuli):
        for col in ('image_label', 'label'):
            if col in stimuli.columns:
                return list(stimuli[col].values)
        raise ValueError(
            f"Cannot find labels in fitting_stimuli. Expected one of "
            f"'image_label' or 'label' columns; got: {list(stimuli.columns)}."
        )

    def extract_behavioral_features(self, stimuli):
        """Run the model once at the behavioral readout layer."""
        owner = self.owner
        saved_layer = owner._recording_layer
        saved_layers = owner._recording_layers
        saved_regions = owner._recording_regions
        saved_multi = owner._is_multi_region
        saved_classifier = self.readout_classifier
        owner._recording_layer = owner._behavioral_readout_layer
        owner._recording_layers = (
            [owner._behavioral_readout_layer]
            if owner._behavioral_readout_layer else []
        )
        owner._recording_regions = []
        owner._is_multi_region = False
        self.readout_classifier = None
        try:
            features = owner.process(stimuli)
        finally:
            owner._recording_layer = saved_layer
            owner._recording_layers = saved_layers
            owner._recording_regions = saved_regions
            owner._is_multi_region = saved_multi
            self.readout_classifier = saved_classifier

        if 'presentation' in features.dims and 'neuroid' in features.dims:
            features = features.transpose('presentation', 'neuroid')
        return features

    def predict_probabilities(self, stimuli):
        """Return a BehavioralAssembly of per-label probabilities."""
        features = self.owner._extract_behavioral_features(stimuli)
        return self.readout_classifier.predict_proba(features)

    def generate_predictions(self, stimuli):
        """Call generation_fn per-stimulus, return one-hot BehavioralAssembly."""
        import numpy as np
        from brainscore_core.supported_data_standards.brainio.assemblies import (
            BehavioralAssembly,
        )

        owner = self.owner
        task_context = self.task_context
        if task_context is None:
            raise RuntimeError(
                "generate_predictions requires an active TaskContext."
            )
        label_set = list(task_context.label_set)
        instruction = task_context.instruction
        label_to_idx = {lbl: i for i, lbl in enumerate(label_set)}

        n_stimuli = len(stimuli)
        n_labels = len(label_set)
        proba = np.zeros((n_stimuli, n_labels), dtype=np.float32)

        for i, (_, row) in enumerate(stimuli.iterrows()):
            predicted = owner._generation_fn(
                stimulus_row=row,
                instruction=instruction,
                label_set=label_set,
            )
            if predicted not in label_to_idx:
                import warnings
                warnings.warn(
                    f"generation_fn returned {predicted!r}, not in "
                    f"label_set={label_set}. Defaulting to {label_set[0]!r}."
                )
                predicted = label_set[0]
            proba[i, label_to_idx[predicted]] = 1.0

        stimulus_ids = list(stimuli['stimulus_id'].values)
        presentation_coords = {
            'stimulus_id': ('presentation', stimulus_ids),
        }
        for column in stimuli.columns:
            if column == 'stimulus_id':
                continue
            presentation_coords[column] = (
                'presentation', list(stimuli[column].values))

        return BehavioralAssembly(
            proba,
            coords={
                **presentation_coords,
                'choice': label_set,
            },
            dims=['presentation', 'choice'],
        )
