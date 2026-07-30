"""Concrete Brain-Score subject implementation."""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .behavioral import BehavioralReadout
from .capability_config import normalize_capability_config
from .contract import Subject, TaskContext
from .dispatch import InputDispatcher, register_input_handler
from .io_catalog import canonical_modality
from .events import (
    EnvironmentStep,
    InputEvent,
    Message,
    OutputEvent,
    StateChange,
)
from .perturbation import PerturbationManager
from .recording import Recorder
from .selection import UnitSelector, _promote_to_selector
from .streaming import StreamEvent
from .streaming_helpers import (
    _drain_stream_events as _drain_neural_stream_events,
    _drive_behavior_session_via_process,
    _drive_environment_session_via_process,
    _drive_neural_session_via_process,
    _drive_state_change_session_via_process,
    _reconstruct_stimulus_set_from_events as _reconstruct_neural_stimulus_set,
    _requested_output_channels,
    _stimuli_from_stream_session as _neural_stimuli_from_stream_session,
)


class BrainScoreModel(Subject):
    """Compositional implementation of Subject (formerly UnifiedModel)."""

    COLUMN_TO_MODALITY: Dict[str, str] = {
        'image_file_name': 'vision',
        'image_path': 'vision',
        'filename': 'vision',
        'sentence': 'text',
        'text': 'text',
        'video_path': 'vision',  # video is temporal vision -> the one visual channel
        'audio_path': 'audio',
        'audio_file_name': 'audio',
        'audio_file': 'audio',
    }

    MODALITY_PRIORITY: Tuple[str, ...] = ('vision', 'text', 'audio')

    _INPUT_HANDLERS: List[Tuple[type, str]] = [
        (StateChange, '_dispatch_state_change'),
        (EnvironmentStep, '_handle_environment_step'),
        (Message, '_handle_message'),
    ]

    BEHAVIORAL_TASK_TYPES = BehavioralReadout.BEHAVIORAL_TASK_TYPES

    def __init__(
        self,
        identifier: str,
        model: Any = None,
        region_layer_map: Optional[Dict[str, Union[str, "UnitSelector"]]] = None,
        preprocessors: Optional[Dict[str, Callable]] = None,
        activations_model: Any = None,
        visual_degrees: int = 8,
        *legacy_tail_args: Any,
        behavioral_readout_layer: Optional[str] = None,
        generation_fn: Optional[Callable] = None,
        action_fn: Optional[Callable] = None,
        state_change_fn: Optional[Callable] = None,
        required_modalities: Optional[Set[str]] = None,
        backbone_id: Optional[str] = None,
        region_modality_map: Optional[Dict[str, str]] = None,
        capability_config: Optional[Dict[str, Any]] = None,
        **legacy_capability_config: Any,
    ) -> None:
        """A model (or, in principle, a human) evaluated through the unified interface.

        Only ``identifier`` is always required; a feature-extraction model also
        needs ``region_layer_map`` plus a ``preprocessors`` / ``activations_model``
        pair. The four capability functions are optional and each enables an extra
        evaluation path.

        :param identifier: unique model name (also the activation-cache key).
        :param model: the object perturbation hooks / generation run against (e.g.
            the torch module); may be ``None`` for a feature-extraction model whose
            wrapper holds its own module.
        :param region_layer_map: brain region -> layer path (or a ``UnitSelector``).
        :param preprocessors: modality -> callable. For ``vision``: a resize/normalize
            callable paired with ``activations_model``. For ``text``/``audio``/``video``:
            the whole extraction wrapper, with ``activations_model=None``. See EXTENDING.
        :param activations_model: the vision extraction wrapper (e.g. PytorchWrapper).
        :param visual_degrees: vision field-of-view; ignored by non-vision models.
        :param behavioral_readout_layer: layer on which to fit a behavioral readout.
        :param generation_fn: generate-and-parse callable for instruction-following
            behavioral evaluation.
        :param action_fn: ``EnvironmentStep -> EnvironmentResponse`` policy for
            closed-loop / embodied evaluation.
        :param state_change_fn: ``StateChange -> (PerturbationApplied, cleanup)`` for
            lesion / stimulation evaluation.
        :param required_modalities: input modalities the model cannot run without.
        :param backbone_id: shared cache key so registrations with identical weights
            reuse one activation cache.
        :param region_modality_map: region -> modality routing for multi-tower models.

        Note: passing capability functions POSITIONALLY (via the legacy tail) is
        deprecated and error-prone; always pass them by keyword.
        """
        if legacy_tail_args:
            import warnings
            warnings.warn(
                "Passing capability functions positionally to BrainScoreModel is "
                "deprecated and error-prone: a mis-counted positional silently lands "
                "in the wrong capability slot. Pass them by keyword instead "
                "(behavioral_readout_layer=, generation_fn=, action_fn=, "
                "state_change_fn=).",
                DeprecationWarning, stacklevel=2)
        # Fold the now-explicit capability kwargs into the config the normalizer
        # sees, so they are visible in the signature/help() yet routed uniformly.
        for _name, _val in (
            ('behavioral_readout_layer', behavioral_readout_layer),
            ('generation_fn', generation_fn),
            ('action_fn', action_fn),
            ('state_change_fn', state_change_fn),
        ):
            if _val is not None:
                legacy_capability_config.setdefault(_name, _val)
        capability_values, required_modalities, backbone_id, region_modality_map = (
            normalize_capability_config(
                capability_config,
                legacy_tail_args,
                legacy_capability_config,
                required_modalities,
                backbone_id,
                region_modality_map,
            )
        )
        self._identifier_str = identifier
        self._model = model
        region_layer_map = region_layer_map or {}
        preprocessors = preprocessors or {}
        self._region_layer_selectors: Dict[str, UnitSelector] = {
            region: _promote_to_selector(value)
            for region, value in region_layer_map.items()
        }
        self._region_layer_map_dict: Dict[str, str] = {
            region: selector.layer_path
            for region, selector in self._region_layer_selectors.items()
        }
        self._region_modality_map: Dict[str, str] = dict(
            region_modality_map or {})
        if self._region_modality_map:
            unknown_regions = (set(self._region_modality_map.keys())
                               - set(region_layer_map.keys()))
            if unknown_regions:
                raise ValueError(
                    f"region_modality_map references regions not in "
                    f"region_layer_map: {sorted(unknown_regions)}."
                )
            unknown_modalities = (
                {canonical_modality(m) for m in self._region_modality_map.values()}
                - {canonical_modality(k) for k in preprocessors.keys()})
            if unknown_modalities:
                raise ValueError(
                    f"region_modality_map references modalities with no "
                    f"preprocessor: {sorted(unknown_modalities)}. Available "
                    f"modalities: {sorted(preprocessors.keys())}."
                )
        self._preprocessors = preprocessors
        self._activations_model = activations_model
        self._visual_degrees_val = visual_degrees
        self._capability_config = capability_values
        self._behavioral_readout_layer = capability_values.get(
            'behavioral_readout_layer')
        self._generation_fn = capability_values.get('generation_fn')
        self._action_fn = capability_values.get('action_fn')
        self._state_change_fn = capability_values.get('state_change_fn')

        self._recorder = Recorder(self)
        self._behavioral = BehavioralReadout(self)
        self._perturbations = PerturbationManager(self)
        self._dispatcher = InputDispatcher(self)
        self._capability_state: Dict[str, Any] = {}
        self._capability_setup_done: Set[str] = set()

        required = ({canonical_modality(m) for m in required_modalities}
                    if required_modalities else set())
        available = {canonical_modality(k) for k in preprocessors.keys()}
        if not required.issubset(available):
            raise ValueError(
                f"required_modalities {required} must be a subset of the "
                f"available preprocessors {available}. A model cannot require "
                f"a modality for which it has no preprocessor."
        )
        self._required_modalities = required
        self._backbone_id = backbone_id or identifier

    @property
    def identifier(self) -> str:
        return self._identifier_str

    @property
    def region_layer_map(self) -> Dict[str, str]:
        return dict(self._region_layer_map_dict)

    @property
    def region_layer_selectors(self) -> Dict[str, "UnitSelector"]:
        return dict(self._region_layer_selectors)

    @property
    def supported_modalities(self) -> Set[str]:
        # a ``video``-keyed preprocessor reports as ``vision`` (temporal vision)
        return {canonical_modality(m) for m in self._preprocessors.keys()}

    @property
    def available_modalities(self) -> Set[str]:
        return {canonical_modality(m) for m in self._preprocessors.keys()}

    @property
    def required_modalities(self) -> Set[str]:
        return set(self._required_modalities)

    @property
    def backbone_id(self) -> str:
        return self._backbone_id

    @property
    def _recording_layer(self) -> Optional[str]:
        return self._recorder.recording_layer

    @_recording_layer.setter
    def _recording_layer(self, value: Optional[str]) -> None:
        self._recorder.recording_layer = value

    @property
    def _recording_regions(self) -> List[str]:
        return self._recorder.recording_regions

    @_recording_regions.setter
    def _recording_regions(self, value: List[str]) -> None:
        self._recorder.recording_regions = value

    @property
    def _recording_layers(self) -> List[str]:
        return self._recorder.recording_layers

    @_recording_layers.setter
    def _recording_layers(self, value: List[str]) -> None:
        self._recorder.recording_layers = value

    @property
    def _is_multi_region(self) -> bool:
        return self._recorder.is_multi_region

    @_is_multi_region.setter
    def _is_multi_region(self, value: bool) -> None:
        self._recorder.is_multi_region = value

    @property
    def _time_bins(self) -> Optional[List[Tuple[int, int]]]:
        return self._recorder.time_bins

    @_time_bins.setter
    def _time_bins(self, value: Optional[List[Tuple[int, int]]]) -> None:
        self._recorder.time_bins = value

    @property
    def _composite_recording(self) -> bool:
        return self._recorder.composite_recording

    @_composite_recording.setter
    def _composite_recording(self, value: bool) -> None:
        self._recorder.composite_recording = value

    @property
    def _task_context(self) -> Optional[TaskContext]:
        return self._behavioral.task_context

    @_task_context.setter
    def _task_context(self, value: Optional[TaskContext]) -> None:
        self._behavioral.task_context = value

    @property
    def _readout_classifier(self):
        return self._behavioral.readout_classifier

    @_readout_classifier.setter
    def _readout_classifier(self, value) -> None:
        self._behavioral.readout_classifier = value

    @property
    def _use_generation_for_task(self) -> bool:
        return self._behavioral.use_generation_for_task

    @_use_generation_for_task.setter
    def _use_generation_for_task(self, value: bool) -> None:
        self._behavioral.use_generation_for_task = value

    @property
    def _active_perturbations(self):
        return self._perturbations.active_perturbations

    @_active_perturbations.setter
    def _active_perturbations(self, value) -> None:
        self._perturbations.active_perturbations = value

    @classmethod
    def register_input_handler(cls, event_type: type, handler_name: str,
                               before: Optional[type] = None) -> None:
        register_input_handler(cls, event_type, handler_name, before)

    def process(self, input_event: InputEvent,
                multi_modality: bool = False) -> OutputEvent:
        """Evaluate one input event and return the corresponding output event.

        The single evaluation entry point; dispatch is by ``input_event`` type:
        a ``StimulusSet`` / ``MultimodalStimulusSet`` -> ``NeuroidAssembly`` of
        recorded activations (call ``start_recording`` first); a ``StateChange``
        -> ``PerturbationApplied`` (apply) or ``None`` (reset); an
        ``EnvironmentStep`` -> ``EnvironmentResponse`` (needs ``action_fn``).

        :param input_event: the stimulus / state-change / environment-step to run.
        :param multi_modality: if True, extract from every detected-and-supported
            modality and concat along the neuroid axis; default routes a single
            modality by ``MODALITY_PRIORITY``.
        """
        return self._dispatcher.process(input_event, multi_modality)

    def interact(self, session) -> None:
        """Drive a v2 streaming session through existing model paths."""
        requested_channels = _requested_output_channels(session)
        if requested_channels == ["behavior"]:
            _drive_behavior_session_via_process(self, session)
            return
        if requested_channels == ["perturbation"]:
            _drive_state_change_session_via_process(self, session)
            return
        if requested_channels == ["motor"]:
            _drive_environment_session_via_process(self, session)
            return
        if all(channel.startswith("neural:") for channel in requested_channels):
            _drive_neural_session_via_process(self, session)
            return
        raise NotImplementedError(
            "BrainScoreModel.interact currently supports requested "
            "neural:<region>, behavior, perturbation, or motor output "
            "channels; got "
            f"{requested_channels!r}."
        )

    def _handle_environment_step(self, input_event: 'EnvironmentStep') -> OutputEvent:
        return self._dispatcher.handle_environment_step(input_event)

    def _handle_message(self, input_event: 'Message') -> OutputEvent:
        return self._dispatcher.handle_message(input_event)

    def _handle_stimulus_set(self, stimuli,
                             multi_modality: bool = False) -> OutputEvent:
        return self._dispatcher.handle_stimulus_set(stimuli, multi_modality)

    def _detect_modalities(self, stimuli) -> Set[str]:
        return self._dispatcher.detect_modalities(stimuli)

    def _pick_modality(self, detected: Set[str]) -> str:
        return self._dispatcher.pick_modality(detected)

    def _filter_layers_for_modality(self, layers: List[str],
                                    modality: str) -> List[str]:
        return self._recorder.filter_layers_for_modality(layers, modality)

    def _supports_layer_extraction(self, modality: str) -> bool:
        return self._dispatcher.supports_layer_extraction(modality)

    def _extract_for_modality(self, stimuli, modality: str, layers: List[str]):
        return self._dispatcher.extract_for_modality(stimuli, modality, layers)

    def _process_multi_modality(self, stimuli, detected: Set[str],
                                layers: List[str]):
        return self._dispatcher.process_multi_modality(stimuli, detected, layers)

    def _tag_neuroids_with_regions(self, assembly):
        return self._recorder.tag_neuroids_with_regions(assembly)

    def _process_composite_regions(self, stimuli, modality):
        return self._recorder.process_composite_regions(stimuli, modality)

    def _gather_composite_units(self, assembly, selector):
        return self._recorder.gather_composite_units(assembly, selector)

    def start_recording(self,
                        recording_target: Union[str, List[str]],
                        time_bins: Optional[List[Tuple[int, int]]] = None,
                        recording_type: Optional[str] = None) -> None:
        self._recorder.start_recording(recording_target, time_bins,
                                       recording_type)

    def start_task(self, task_context, fitting_stimuli=None,
                   **kwargs) -> None:
        self._behavioral.start_task(task_context, fitting_stimuli,
                                    **kwargs)

    def reset(self) -> None:
        self._recorder.reset()
        self._behavioral.reset()
        self._perturbations.reset()

    def _resolve_selection(self, state_change: 'StateChange') -> 'StateChange':
        return self._perturbations.resolve_selection(state_change)

    def _dispatch_state_change(self, state_change: 'StateChange') -> Any:
        return self._dispatcher.dispatch_state_change(state_change)

    @staticmethod
    def _drain_stream_events(session) -> List[StreamEvent]:
        return _drain_neural_stream_events(session)

    @staticmethod
    def _stimuli_from_stream_session(session, events: List[StreamEvent]):
        return _neural_stimuli_from_stream_session(session, events)

    @staticmethod
    def _reconstruct_stimulus_set_from_events(events: List[StreamEvent]):
        """Rebuild a minimal StimulusSet from event-carried input columns.

        This fallback preserves only input-channel payload columns plus
        ``stimulus_id``. It is not metadata-faithful: metric-critical
        presentation metadata such as ``object_name`` must come from
        session.stimulus_set, or from a future event-carried metadata path.
        """
        return _reconstruct_neural_stimulus_set(events)

    @staticmethod
    def _requested_output_channels(session) -> List[str]:
        return _requested_output_channels(session)

    @classmethod
    def _requires_behavioral_readout(cls, task_type: str) -> bool:
        return BehavioralReadout.requires_behavioral_readout(task_type)

    def _fit_behavioral_readout(self, fitting_stimuli) -> None:
        self._behavioral.fit_behavioral_readout(fitting_stimuli)

    @staticmethod
    def _extract_labels(stimuli):
        return BehavioralReadout.extract_labels(stimuli)

    def _extract_behavioral_features(self, stimuli):
        return self._behavioral.extract_behavioral_features(stimuli)

    def _predict_probabilities(self, stimuli):
        return self._behavioral.predict_probabilities(stimuli)

    def _generate_predictions(self, stimuli):
        return self._behavioral.generate_predictions(stimuli)

    def look_at(self, stimuli, number_of_trials=1, **kwargs):
        """Vision benchmark compatibility. Delegates to process()."""
        return self.process(stimuli)

    def visual_degrees(self) -> int:
        """Vision benchmark compatibility."""
        return self._visual_degrees_val

    def digest_text(self, text) -> Dict[str, Any]:
        """Language benchmark compatibility."""
        import hashlib
        import pandas as pd
        from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

        import numpy as np
        if isinstance(text, np.ndarray):
            text = text.tolist()
        if isinstance(text, (str, list)):
            if isinstance(text, str):
                text = [text]
            stimuli = StimulusSet(pd.DataFrame({
                'sentence': text,
                'stimulus_id': list(range(len(text))),
            }))
            content_hash = hashlib.md5(
                '|'.join(text).encode('utf-8')).hexdigest()[:12]
            stimuli.identifier = f'text_stimuli_{len(text)}_{content_hash}'
        else:
            stimuli = text

        result = self.process(stimuli)
        return {'neural': result}

    def start_neural_recording(self, recording_target, recording_type='fMRI'):
        """Language benchmark compatibility."""
        self.start_recording(recording_target, recording_type=recording_type)
