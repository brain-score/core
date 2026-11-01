"""Concrete Brain-Score subject implementation."""

from copy import copy
from dataclasses import replace
from types import MappingProxyType
from typing import Mapping, Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .behavioral import BehavioralReadout
from .capability_config import normalize_capability_config
from .contract import UnifiedModel, TaskContext
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
from .selection import CompositeSelector, LayerSelector, UnitSelector, _promote_to_selector
from .streaming import StreamEvent
from .streaming_helpers import (
    _drain_stream_events as _drain_neural_stream_events,
    _reconstruct_stimulus_set_from_events as _reconstruct_neural_stimulus_set,
    _requested_output_channels,
    _stimuli_from_stream_session as _neural_stimuli_from_stream_session,
)


_PREFERRED_NAMES = {
    'preprocessors': "extractor={modality: extractor} (configure input preparation "
                     "on the extractor; declare extractor-less inputs with "
                     "supported_modalities=)",
    'activations_model': 'extractor=',
    'generation_fn': 'answer=',
    'action_fn': 'policy=',
    'state_change_fn': 'intervention=',
}


def _warn_old_names(**given) -> None:
    """Deprecation warning for each older constructor name that was used."""
    import warnings
    for name, value in given.items():
        if value is None or (name == 'preprocessors' and not value):
            continue
        warnings.warn(
            f"BrainScoreModel({name}=...) is deprecated; use "
            f"{_PREFERRED_NAMES[name]}. It keeps working.",
            DeprecationWarning, stacklevel=3)


class BrainScoreModel(UnifiedModel):
    """The one concrete subject, assembled from parts (extractors, region map,
    answer, policy, intervention, capabilities). Evaluated through ``interact``;
    ``process`` is a typed convenience and the legacy benchmark methods forward
    to :class:`~brainscore_core.legacy_facade.LegacyBenchmarkFacade`."""

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
        extractor: Any = None,
        answer: Optional[Callable] = None,
        policy: Optional[Callable] = None,
        intervention: Optional[Callable] = None,
        behavioral_readout_layer: Optional[str] = None,
        generation_fn: Optional[Callable] = None,
        action_fn: Optional[Callable] = None,
        state_change_fn: Optional[Callable] = None,
        supported_modalities: Optional[Set[str]] = None,
        required_modalities: Optional[Set[str]] = None,
        backbone_id: Optional[str] = None,
        region_modality_map: Optional[Dict[str, str]] = None,
        capability_config: Optional[Dict[str, Any]] = None,
        **legacy_capability_config: Any,
    ) -> None:
        """A model (or, in principle, a human) evaluated through the unified interface.

        Only ``identifier`` is always required; a feature-extraction model also
        needs ``region_layer_map`` and an ``activations_model`` mapping.
        Capability functions are optional; each enables another evaluation path.

        :param identifier: unique model name (also the activation-cache key).
        :param model: the object perturbation hooks / generation run against (e.g.
            the torch module); may be ``None`` for a feature-extraction model whose
            wrapper holds its own module.
        :param region_layer_map: brain region -> layer path (or a ``UnitSelector``).
        :param preprocessors: input preparation callables, declared by modality.
            Configure each wrapper with its preparation function or tokenizer.
        :param activations_model: extraction callable ``(stimuli, layers=...)``,
            shared across declared modalities, or a modality -> extractor mapping.
            Image, text, audio and video wrappers use this same slot. A mapping
            declares its modalities without needing placeholder preprocessors.
        :param visual_degrees: vision field-of-view; ignored by non-vision models.
        :param behavioral_readout_layer: layer on which to fit a behavioral readout.
        :param extractor: extraction callable or modality-to-extractor mapping.
            Prepare inputs inside each extractor. Preferred over activations_model.
        :param answer: generate and parse a behavioral answer.
        :param policy: choose an action from an EnvironmentStep.
        :param intervention: apply a StateChange and return its cleanup function.
            The return value is (PerturbationApplied, cleanup).
        :param generation_fn: generate-and-parse callable for instruction-following
            behavioral evaluation.
        :param action_fn: ``EnvironmentStep -> EnvironmentResponse`` policy for
            closed-loop / embodied evaluation.
        :param state_change_fn: ``StateChange -> (PerturbationApplied, cleanup)`` for
            lesion / stimulation evaluation.
        :param supported_modalities: inputs accepted without an extraction wrapper,
            for example by a closed-weight generation function. Extractor mappings
            already declare their inputs; do not repeat them here.
        :param required_modalities: input modalities the model cannot run without.
        :param backbone_id: shared cache key so registrations with identical weights
            reuse one activation cache.
        :param region_modality_map: compatibility routing map. Prefer the modality
            field on LayerSelector or CompositeSelector in region_layer_map.

        Note: passing capability functions POSITIONALLY (via the legacy tail) is
        deprecated and error-prone; always pass them by keyword.
        """
        # One name per part (Contract Rules v3): extractor, answer, policy,
        # intervention. The older names keep working with a deprecation warning.
        _warn_old_names(preprocessors=preprocessors,
                        activations_model=activations_model,
                        generation_fn=generation_fn, action_fn=action_fn,
                        state_change_fn=state_change_fn,
                        **{k: legacy_capability_config[k] for k in
                           ('generation_fn', 'action_fn', 'state_change_fn')
                           if k in legacy_capability_config})
        # Translate the public vocabulary once; existing plugins keep their
        # callback names and the internal capability contract stays unchanged.
        if extractor is not None:
            if activations_model is not None:
                raise TypeError("Use extractor or activations_model, not both")
            activations_model = extractor
        for public_name, value, old_name, old_value in (
            ('answer', answer, 'generation_fn', generation_fn),
            ('policy', policy, 'action_fn', action_fn),
            ('intervention', intervention, 'state_change_fn', state_change_fn),
        ):
            if value is not None:
                if old_value is not None or old_name in legacy_capability_config:
                    raise TypeError(f"Use {public_name} or {old_name}, not both")
                legacy_capability_config[old_name] = value
        if legacy_tail_args:
            import warnings
            warnings.warn(
                "Passing capability functions positionally to BrainScoreModel is "
                "deprecated and error-prone: a mis-counted positional silently lands "
                "in the wrong capability slot. Pass them by keyword instead "
                "(behavioral_readout_layer=, answer=, policy=, intervention=).",
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
        # Normalize compatibility inputs once; dispatch always calls an extractor.
        from .dispatch import normalize_extractors
        self._preprocessors: Dict[str, Optional[Callable]] = dict(preprocessors)
        self._activations_model = activations_model
        self._extractors, self._layer_extractors = normalize_extractors(
            self, preprocessors, activations_model)
        for modality in supported_modalities or ():
            self._preprocessors.setdefault(modality, None)
        for modality in self._extractors:
            self._preprocessors.setdefault(modality, None)
        self._region_modality_map: Dict[str, str] = {}
        self._install_region_layer_map(region_layer_map, region_modality_map)
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
        available = self.supported_modalities
        if not required.issubset(available):
            raise ValueError(
                f"required_modalities {required} must be a subset of the "
                f"supported modalities {available}. A model cannot require "
                f"an input it does not declare."
        )
        self._required_modalities = required
        self._backbone_id = backbone_id or identifier

    @property
    def capability_config(self):
        """Read-only registration configuration for external capability authors."""
        return MappingProxyType(self._capability_config)

    def capability_state(self, identifier):
        """Per-model capability state, initialized by the registry before dispatch."""
        return self._capability_state[identifier]

    def _declared_capabilities(self):
        from .capabilities import capability_registry
        return [c for c in capability_registry.values() if c.enabled_for(self)]

    @property
    def in_channels(self):
        return super().in_channels | set().union(
            *(c.declared_input_channels(self) for c in self._declared_capabilities()))

    @property
    def out_channels(self):
        # Every output comes from a registered part; nothing is found by name.
        return set().union(
            *(c.declared_output_channels(self) for c in self._declared_capabilities()))

    def check_session_support(self, channels):
        """The one capability that runs a session requesting ``channels``.

        Raises before any input is consumed when no capability, or more than
        one at the same priority, claims the combination.
        """
        channels = frozenset(channels)
        if not channels:
            raise ValueError('A session must request at least one output channel.')
        claims = [c for c in self._declared_capabilities()
                  if c.supports_session(self, channels)]
        if not claims:
            raise NotImplementedError(f'No session handler for {sorted(channels)}')
        first = min(c.order for c in claims)
        winners = [c for c in claims if c.order == first]
        if len(winners) > 1:
            raise ValueError(f'Ambiguous session handlers: {[c.identifier for c in winners]}')
        return winners[0]

    @property
    def identifier(self) -> str:
        return self._identifier_str

    def _install_region_layer_map(self, region_layer_map, routing=None) -> None:
        """Validate and install one lossless region configuration atomically."""
        selectors = {region: _promote_to_selector(value)
                     for region, value in dict(region_layer_map).items()}
        legacy_routing = (dict(routing) if routing is not None else
                          {r: m for r, m in self._region_modality_map.items()
                           if r in selectors})
        unknown = set(legacy_routing) - set(selectors)
        if unknown:
            raise ValueError(f"region_modality_map references regions not in region_layer_map: {sorted(unknown)}")
        for region, selector in selectors.items():
            modality = getattr(selector, 'modality', None)
            old_modality = legacy_routing.get(region)
            if routing is not None and modality and old_modality and (
                    canonical_modality(modality) != canonical_modality(old_modality)):
                raise ValueError(f"Conflicting modality routing for region {region!r}")
            modality = modality or old_modality
            if modality is not None:
                if canonical_modality(modality) not in self.supported_modalities:
                    raise ValueError(f"Region {region!r} references modalities with no preprocessor or extractor: {modality!r}")
                if isinstance(selector, (LayerSelector, CompositeSelector)):
                    selectors[region] = replace(selector, modality=modality)
                elif selector.modality != modality:
                    # External selectors need only the UnitSelector contract,
                    # not a dataclass constructor accepting our optional field.
                    routed = copy(selector)
                    routed.modality = modality
                    selectors[region] = routed
        self._region_layer_selectors = selectors
        self._region_layer_map_dict = {r: sel.layer_path for r, sel in selectors.items()}
        self._region_modality_map = {
            r: sel.modality for r, sel in selectors.items()
            if sel.modality is not None}

    @property
    def region_layer_map(self) -> Mapping[str, Union[str, UnitSelector]]:
        """Read-only, lossless recording configuration.

        Simple entries remain layer strings. Composite units and modality routing
        remain selectors, so assigning this mapping back preserves every setting.
        """
        return MappingProxyType({
            region: (selector.name if isinstance(selector, LayerSelector)
                     and selector.modality is None else selector)
            for region, selector in self._region_layer_selectors.items()})

    @region_layer_map.setter
    def region_layer_map(self, region_layer_map) -> None:
        self._install_region_layer_map(region_layer_map)

    def set_region_layer(self, region: str, value) -> None:
        """Replace one region's layer or selector, keeping its existing routing."""
        updated = dict(self._region_layer_selectors)
        updated[region] = value
        self._install_region_layer_map(updated)

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
    def input_modalities(self) -> Set[str]:
        """The RAW declared input-modality keys (pre-canonicalization).

        ``supported_modalities`` canonicalizes ``video`` into ``vision`` (temporal
        vision) so compatibility and routing treat them as one modality. This
        property keeps the *declared processing style* instead: a native-video
        model reports ``'video'`` here (its VideoWrapper consumes whole clips) but
        ``'vision'`` in ``supported_modalities``. Benchmarks that must present
        whole clips vs. still frames route on this, not on ``supported_modalities``.
        """
        return set(self._preprocessors.keys())

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

        A typed convenience, not part of the subject contract (``interact`` is
        the evaluation method). Dispatch is by ``input_event`` type:
        a ``StimulusSet`` / ``MultimodalStimulusSet`` -> ``NeuroidAssembly`` of
        recorded activations (call ``start_recording`` first); a ``StateChange``
        -> ``PerturbationApplied`` (apply) or ``None`` (reset); an
        ``EnvironmentStep`` -> ``EnvironmentResponse`` (needs ``action_fn``).

        :param input_event: the stimulus / state-change / environment-step to run.
        :param multi_modality: if True, extract from every detected-and-supported
            modality and concat along the neuroid axis; default routes a single
            modality by ``MODALITY_PRIORITY``.
        """
        output = self._dispatcher.process(input_event, multi_modality)
        time_bins = self._recorder.time_bins
        if time_bins is not None and len(time_bins) > 1 and not isinstance(
                input_event, (StateChange, EnvironmentStep, Message)):
            # Honor every requested time bin: a static response is repeated
            # across several bins, as legacy vision models do.
            from .legacy_facade import repeat_over_time_bins
            output = repeat_over_time_bins(output, time_bins)
        return output

    def interact(self, session) -> None:
        """Run a session through the one registered capability that claims it."""
        handler = self.check_session_support(_requested_output_channels(session))
        from .capabilities.registry import _setup_once
        _setup_once(self, handler)
        handler.interact(self, session)

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

    def _tag_neuroids_with_regions(self, assembly, modality=None):
        return self._recorder.tag_neuroids_with_regions(assembly, modality)

    def _process_composite_regions(self, stimuli, modality):
        return self._recorder.process_composite_regions(stimuli, modality)

    def start_recording(self,
                        recording_target: Union[str, List[str]],
                        time_bins: Optional[List[Tuple[int, int]]] = None,
                        recording_type: Optional[str] = None) -> None:
        """Configure neural recording for subsequent ``process`` calls.

        ``recording_target`` is a mapped region name, ``'all'``, a list of
        mapped regions, or a raw layer path. Multiple regions are extracted in
        one call and returned with per-neuroid ``layer`` and ``region``
        provenance. See :class:`Recorder` for validation details.
        """
        self._recorder.start_recording(recording_target, time_bins,
                                       recording_type)

    def start_task(self, task_context, fitting_stimuli=None,
                   **kwargs) -> None:
        self._behavioral.start_task(task_context, fitting_stimuli,
                                    **kwargs)

    def reset(self) -> None:
        failures = []
        for component in (self._recorder, self._behavioral, self._perturbations):
            try:
                component.reset()
            except Exception as error:
                failures.append(error)
        # Callable capabilities and extractors may own episode/history state.
        # Reset each provider once, including bound-method registrations.
        providers = [self._action_fn, self._generation_fn, self._activations_model,
                     *self._preprocessors.values(), *self._extractors.values()]
        seen = set()
        from .capabilities import capability_registry
        for identifier, state in self._capability_state.items():
            capability = capability_registry.get(identifier)
            if capability is not None:
                try:
                    capability.reset(self, state)
                    self._capability_setup_done.discard(identifier)
                except Exception as error:
                    failures.append(error)
        for provider in providers:
            provider = getattr(provider, '__self__', provider)
            if provider is None or provider is self or id(provider) in seen:
                continue
            seen.add(id(provider))
            reset = getattr(provider, 'reset', None)
            if callable(reset):
                try:
                    reset()
                except Exception as error:
                    failures.append(error)
        if failures:
            raise RuntimeError('A model provider failed to reset; the model is not clean.') from failures[0]

    def _dispatch_state_change(self, state_change: 'StateChange') -> Any:
        return self._dispatcher.dispatch_state_change(state_change)

    @staticmethod
    def _reconstruct_stimulus_set_from_events(events: List[StreamEvent]):
        """Rebuild a minimal StimulusSet from event-carried input columns.

        This fallback preserves only input-channel payload columns plus
        ``stimulus_id``. It is not metadata-faithful: metric-critical
        presentation metadata such as ``object_name`` must come from
        session.stimulus_set, or from a future event-carried metadata path.
        """
        return _reconstruct_neural_stimulus_set(events)

    @classmethod
    def _requires_behavioral_readout(cls, task_type: str) -> bool:
        return BehavioralReadout.requires_behavioral_readout(task_type)

    def _fit_behavioral_readout(self, fitting_stimuli) -> None:
        self._behavioral.fit_behavioral_readout(fitting_stimuli)

    def _extract_behavioral_features(self, stimuli):
        return self._behavioral.extract_behavioral_features(stimuli)

    def _predict_probabilities(self, stimuli):
        return self._behavioral.predict_probabilities(stimuli)

    def _generate_predictions(self, stimuli):
        return self._behavioral.generate_predictions(stimuli)

    # --- Legacy benchmark surface ------------------------------------------
    # Permanent: legacy benchmarks are never removed. Behavior lives in
    # LegacyBenchmarkFacade, which runs each legacy call as a session through
    # ``interact``; these forwards seed a facade with this model's current
    # recording and task state so direct calls keep working.

    @property
    def _legacy_facade(self):
        from .legacy_facade import LegacyBenchmarkFacade
        facade = LegacyBenchmarkFacade(self, visual_degrees=self._visual_degrees_val)
        regions = self._recorder.recording_regions
        if regions:
            facade.start_recording(regions[0] if len(regions) == 1 else list(regions),
                                   time_bins=self._recorder.time_bins)
        if self._behavioral.task_context is not None:
            facade.start_task(self._behavioral.task_context)
        return facade

    def look_at(self, stimuli, number_of_trials=1, require_variance=False, **kwargs):
        """Vision benchmark compatibility (see LegacyBenchmarkFacade.look_at)."""
        return self._legacy_facade.look_at(stimuli, number_of_trials,
                                           require_variance, **kwargs)

    def visual_degrees(self) -> int:
        """Vision benchmark compatibility."""
        return self._visual_degrees_val

    def digest_text(self, text) -> Dict[str, Any]:
        """Language benchmark compatibility (see LegacyBenchmarkFacade.digest_text)."""
        return self._legacy_facade.digest_text(text)

    def start_neural_recording(self, recording_target, recording_type='fMRI'):
        """Language benchmark compatibility."""
        self.start_recording(recording_target, recording_type=recording_type)

    def start_behavioral_task(self, task):
        """Language benchmark compatibility; typed models have no legacy language task path."""
        raise NotImplementedError(
            f"Subject {self.identifier!r} has no path for the legacy language "
            f"behavioral task {getattr(task, 'value', task)!r} (reading times / "
            f"next-word prediction). Legacy language behavioral benchmarks need "
            f"a subject that implements that task.")
