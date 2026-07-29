"""Input-event dispatch and modality extraction for BrainScoreModel."""

import os
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional, Set, Tuple

from .capabilities import enabled_capabilities
from .events import EnvironmentStep, Message, OutputEvent, StateChange
from .io_catalog import canonical_modality


_cache_disable_depth = 0
_cache_disable_prev = None
_cache_disable_lock = threading.Lock()


@contextmanager
def _activation_cache_disabled():
    """Bypass result_caching for the duration. Used when a perturbation makes the
    cached (unperturbed) activations invalid but the cache key cannot see it.

    Reentrancy- and thread-safe via a lock-guarded depth counter: nested,
    interleaved, or concurrent contexts only restore the original
    ``RESULTCACHING_DISABLE`` when the outermost/last one exits (a plain
    save/restore would clear the flag early; an unlocked counter would race)."""
    global _cache_disable_depth, _cache_disable_prev
    with _cache_disable_lock:
        if _cache_disable_depth == 0:
            _cache_disable_prev = os.environ.get('RESULTCACHING_DISABLE')
            os.environ['RESULTCACHING_DISABLE'] = '1'
        _cache_disable_depth += 1
    try:
        yield
    finally:
        with _cache_disable_lock:
            _cache_disable_depth -= 1
            if _cache_disable_depth == 0:
                if _cache_disable_prev is None:
                    os.environ.pop('RESULTCACHING_DISABLE', None)
                else:
                    os.environ['RESULTCACHING_DISABLE'] = _cache_disable_prev


def register_input_handler(cls, event_type: type, handler_name: str,
                           before: Optional[type] = None) -> None:
    """Register a handler for an input-event type on ``cls``."""
    if '_INPUT_HANDLERS' not in cls.__dict__:
        cls._INPUT_HANDLERS = list(cls._INPUT_HANDLERS)
    entry = (event_type, handler_name)
    if before is not None:
        for i, (existing_type, _) in enumerate(cls._INPUT_HANDLERS):
            if existing_type is before:
                cls._INPUT_HANDLERS.insert(i, entry)
                return
    cls._INPUT_HANDLERS.append(entry)


class InputDispatcher:
    """Routes BrainScoreModel input events to the matching implementation."""

    def __init__(self, owner) -> None:
        self.owner = owner

    def process(self, input_event, multi_modality: bool = False) -> OutputEvent:
        owner = self.owner
        for event_type, handler_name in owner._INPUT_HANDLERS:
            if isinstance(input_event, event_type):
                return getattr(owner, handler_name)(input_event)
        return owner._handle_stimulus_set(input_event, multi_modality)

    def dispatch_state_change(self, input_event: 'StateChange') -> OutputEvent:
        return self.process_capabilities(input_event)

    def handle_environment_step(self, input_event: 'EnvironmentStep') -> OutputEvent:
        return self.process_capabilities(input_event)

    def handle_message(self, input_event: 'Message') -> OutputEvent:
        return self.process_capabilities(input_event)

    def handle_stimulus_set(self, stimuli,
                            multi_modality: bool = False) -> OutputEvent:
        return self.process_capabilities(stimuli, multi_modality)

    def process_capabilities(self, input_event,
                             multi_modality: bool = False) -> OutputEvent:
        owner = self.owner
        for capability in enabled_capabilities(owner):
            if capability.handles(owner, input_event,
                                  multi_modality=multi_modality):
                return capability.process(owner, input_event,
                                          multi_modality=multi_modality)
        raise TypeError(
            f"Model '{owner.identifier}' has no registered capability for "
            f"input event type {type(input_event).__name__}."
        )

    def _preprocessor_for(self, modality: str):
        """The preprocessor whose canonical modality matches ``modality``.

        Resolves a legacy ``video``-keyed preprocessor when requested under the
        canonical ``vision`` name (video is temporal vision). Raises ``KeyError``
        when no key canonicalizes to ``modality``.
        """
        owner = self.owner
        if modality in owner._preprocessors:
            return owner._preprocessors[modality]
        for key, value in owner._preprocessors.items():
            if canonical_modality(key) == modality:
                return value
        raise KeyError(modality)

    def _has_preprocessor_for(self, modality: str) -> bool:
        try:
            self._preprocessor_for(modality)
            return True
        except KeyError:
            return False

    def detect_modalities(self, stimuli) -> Set[str]:
        owner = self.owner
        detected: Set[str] = set()
        for col in stimuli.columns:
            modality = owner.COLUMN_TO_MODALITY.get(col)
            if modality is None:
                continue
            modality = canonical_modality(modality)
            if self._has_preprocessor_for(modality):
                detected.add(modality)
        return detected

    def pick_modality(self, detected: Set[str]) -> str:
        owner = self.owner
        for m in owner.MODALITY_PRIORITY:
            if m in detected:
                return m
        return next(iter(detected))

    def supports_layer_extraction(self, modality: str) -> bool:
        owner = self.owner
        modality = canonical_modality(modality)
        if modality == 'vision' and owner._activations_model is not None:
            return True
        if not self._has_preprocessor_for(modality):
            return False
        return hasattr(self._preprocessor_for(modality), 'identifier')

    def extract_for_modality(self, stimuli, modality: str, layers: List[str]):
        # A perturbation (lesion/stimulation) is a mutable forward-hook that the
        # activation cache key does not fingerprint, so a cached UNperturbed
        # extraction would be wrongly returned for a perturbed run. Bypass the
        # cache while any perturbation is active.
        if self.owner._active_perturbations:
            with _activation_cache_disabled():
                return self._extract_for_modality_inner(stimuli, modality, layers)
        return self._extract_for_modality_inner(stimuli, modality, layers)

    def _extract_for_modality_inner(self, stimuli, modality: str, layers: List[str]):
        owner = self.owner
        modality = canonical_modality(modality)
        if modality == 'vision' and owner._activations_model is not None:
            return owner._activations_model(stimuli, layers=layers)

        preprocessor = self._preprocessor_for(modality)
        if hasattr(preprocessor, 'identifier'):
            return preprocessor(stimuli, layers=layers)
        return preprocessor(
            owner._model, stimuli,
            recording_layer=owner._recording_layer,
        )

    def process_multi_modality(self, stimuli, detected: Set[str],
                               layers: List[str]):
        """Invoke every detected-and-supported wrapper and concat assemblies."""
        import numpy as np
        import xarray as xr

        owner = self.owner
        ordered = sorted(
            detected,
            key=lambda m: (
                owner.MODALITY_PRIORITY.index(m)
                if m in owner.MODALITY_PRIORITY
                else len(owner.MODALITY_PRIORITY)
            ),
        )

        sub_assemblies = []
        for modality in ordered:
            if not self._has_preprocessor_for(modality):
                continue
            modality_layers = owner._filter_layers_for_modality(
                layers, modality)
            if (owner._region_modality_map
                    and not modality_layers
                    and owner._recording_regions):
                continue
            sub = owner._extract_for_modality(stimuli, modality, modality_layers)
            if not hasattr(sub, 'dims') or 'neuroid' not in sub.dims:
                raise TypeError(
                    f"multi_modality dispatch requires every preprocessor "
                    f"to return an xarray-shaped assembly with a 'neuroid' "
                    f"dim; modality '{modality}' returned "
                    f"{type(sub).__name__}. Upgrade the preprocessor to a "
                    f"layer-aware extractor (TextWrapper / VLMVisionWrapper)."
                )
            if owner._is_multi_region:
                sub = owner._tag_neuroids_with_regions(sub)
            n_neuroid = sub.sizes['neuroid']
            sub = sub.assign_coords(
                modality=('neuroid', np.array([modality] * n_neuroid))
            )
            sub_assemblies.append(sub)

        if not sub_assemblies:
            raise ValueError(
                f"multi_modality=True but no detected modalities had a "
                f"matching preprocessor. detected={detected}, "
                f"available={set(owner._preprocessors.keys())}."
            )

        return _concat_modalities_along_neuroid(sub_assemblies, xr)


def _concat_modalities_along_neuroid(sub_assemblies, xr):
    """Concat modality assemblies without letting presentation indexes collide."""
    flattened = [
        _reset_presentation_index(assembly)
        for assembly in sub_assemblies
    ]
    combined = xr.concat(flattened, dim='neuroid')
    presentation_coords = [
        name for name, coord in combined.coords.items()
        if name != 'presentation' and coord.dims == ('presentation',)
    ]
    if presentation_coords:
        combined = combined.set_index(presentation=presentation_coords)
    return combined


def _reset_presentation_index(assembly):
    """Expose presentation MultiIndex levels as plain coords for concat."""
    if 'presentation' not in getattr(assembly, 'indexes', {}):
        return assembly
    index = assembly.indexes['presentation']
    if index.__class__.__name__ != 'MultiIndex':
        return assembly
    return assembly.reset_index('presentation')
