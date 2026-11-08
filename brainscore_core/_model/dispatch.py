"""Input-event dispatch and modality extraction for BrainScoreModel."""

import os
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional, Set, Tuple

from ..capabilities import enabled_capabilities
from ..events import EnvironmentStep, Message, OutputEvent, StateChange
from ..streaming import StreamEvent
from ..io_catalog import canonical_modality


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
    """Compatibility hook; new integrations should register a Capability."""
    import warnings
    warnings.warn(
        'register_input_handler is deprecated; register a Capability instead.',
        DeprecationWarning, stacklevel=3)
    if '_INPUT_HANDLERS' not in cls.__dict__:
        cls._INPUT_HANDLERS = list(cls._INPUT_HANDLERS)
    entry = (event_type, handler_name)
    if before is not None:
        for i, (existing_type, _) in enumerate(cls._INPUT_HANDLERS):
            if existing_type is before:
                cls._INPUT_HANDLERS.insert(i, entry)
                return
    cls._INPUT_HANDLERS.append(entry)


CLIP_COLUMNS = ('video_path', 'video_file_name')


def _is_clip(stimuli) -> bool:
    """Whether a stimulus table presents clips (video) rather than stills."""
    return any(column in getattr(stimuli, 'columns', ()) for column in CLIP_COLUMNS)


def fused_for(owner, detected):
    """The fused extractor taking every detected input type, if one is registered."""
    if len(detected) < 2:
        return None
    for types, extractor in owner._fused_extractors:
        if set(detected) <= types:
            return extractor
    return None


def extractor_for(owner, modality: str, stimuli):
    """The extractor for ``modality``; with a still and a clip extractor, the one
    matching the stimulus's form. None when the model has no such extractor."""
    modality = canonical_modality(modality)
    matching = [key for key in owner._extractors if canonical_modality(key) == modality]
    if len(matching) > 1 and modality == 'vision':
        wanted = 'video' if _is_clip(stimuli) else 'vision'
        matching = [key for key in matching if key == wanted] or matching
    return owner._extractors[matching[0]] if matching else None


def normalize_extractors(owner, preprocessors, activations_model):
    """Adapt registration forms once to ``extractor(stimuli, layers=...)``.

    ``activations_model`` here is the normalized ``extractor`` argument (the
    compatibility spelling is the same slot). Explicit extractors never use
    attribute-based capability discovery. Older registrations that put
    extraction functions in preprocessors are adapted by their callable
    signature at this compatibility boundary.
    """
    from collections.abc import Mapping
    import inspect

    if isinstance(activations_model, Mapping):
        # A fused extractor is registered under every input type it takes at
        # once, {('vision', 'text'): fused}. It also serves each type alone
        # unless a single-type extractor is registered for it.
        fused = [(frozenset(canonical_modality(m) for m in key), value)
                 for key, value in activations_model.items() if isinstance(key, tuple)]
        extractors = {}
        for types, value in fused:
            for modality in types:
                extractors.setdefault(modality, value)
        extractors.update({key: value for key, value in activations_model.items()
                           if not isinstance(key, tuple)})
        owner._fused_extractors = fused
    elif activations_model is not None:
        if not preprocessors:
            raise ValueError(
                'A single shared extractor needs its input modalities declared. '
                'Pass extractor={modality: extractor} instead.')
        extractors = {key: activations_model for key in preprocessors}
    else:
        extractors = {}
    if any(not callable(value) for value in extractors.values()):
        raise TypeError(
            'Every extractor must be callable as extractor(stimuli, layers=...). '
            'Check the values passed in extractor={modality: extractor}.')
    layer_extractors = set(extractors)
    for modality, preprocessor in preprocessors.items():
        if modality in extractors:
            # Historically the single wrapper applied only to vision; an
            # extractor supplied in another modality takes precedence there.
            if isinstance(activations_model, Mapping) or canonical_modality(modality) == 'vision':
                continue
        if not callable(preprocessor):
            continue
        try:
            signature = inspect.signature(preprocessor)
        except (TypeError, ValueError) as error:
            if modality in extractors:
                continue
            raise TypeError(
                'Cannot inspect this preprocessor. Declare it explicitly as '
                'extractor={modality: extractor}.') from error
        variadic_wrapper = (not inspect.isfunction(preprocessor) and
                            bool(signature.parameters) and all(
                                p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                                for p in signature.parameters.values()))
        if 'layers' in signature.parameters or variadic_wrapper:
            extractors[modality] = preprocessor
            layer_extractors.add(modality)
        else:
            try:
                signature.bind(owner._model, object(), recording_layer=None)
                accepts_legacy_call = True
            except TypeError:
                accepts_legacy_call = False
            if modality in extractors and not accepts_legacy_call:
                continue
            import warnings
            warnings.warn(
                'Extraction callbacks in preprocessors are deprecated and are removed at '
                'the public release. Use '
                'extractor={modality: extractor}, where extractor accepts '
                '(stimuli, layers=...).',
                DeprecationWarning, stacklevel=3)
            def extract(stimuli, *, layers, function=preprocessor):
                if len(layers) > 1:
                    raise ValueError(
                        'This compatibility extraction callback accepts one layer. '
                        'For multiple layers, use extractor={modality: '
                        'extractor} with an extractor accepting (stimuli, layers=...).')
                return function(owner._model, stimuli,
                                recording_layer=layers[0] if layers else None)
            extractors[modality] = extract
            layer_extractors.discard(modality)
    return extractors, layer_extractors


class InputDispatcher:
    """Routes BrainScoreModel input events to the matching implementation."""

    def __init__(self, owner) -> None:
        self.owner = owner

    def process(self, input_event, multi_modality: bool = False) -> OutputEvent:
        owner = self.owner
        if isinstance(input_event, StreamEvent):
            from ..io_catalog import validate, INPUT, OUTPUT
            issues = validate(input_event.channel, input_event.payload, INPUT)
            if issues:
                raise ValueError('; '.join(issues))
            output = self.process_capabilities(input_event, multi_modality)
            if isinstance(output, StreamEvent):
                issues = validate(output.channel, output.payload, OUTPUT)
                if issues:
                    raise ValueError('; '.join(issues))
            return output
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
        from ..io_catalog import stimulus_columns
        columns = {**owner.COLUMN_TO_MODALITY, **stimulus_columns()}
        for col in stimuli.columns:
            modality = columns.get(col)
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
        return sorted(detected)[0]

    def supports_layer_extraction(self, modality: str) -> bool:
        return any(canonical_modality(key) == canonical_modality(modality)
                   for key in self.owner._layer_extractors)

    def extract_for_modality(self, stimuli, modality: str, layers: List[str]):
        # A perturbation (lesion/stimulation) is a mutable forward-hook that the
        # activation cache key does not fingerprint, so a cached UNperturbed
        # extraction would be wrongly returned for a perturbed run. Bypass the
        # cache while any perturbation is active.
        if self.owner._perturbations.active_perturbations:
            with _activation_cache_disabled():
                return self._extract_for_modality_inner(stimuli, modality, layers)
        return self._extract_for_modality_inner(stimuli, modality, layers)

    def _extract_for_modality_inner(self, stimuli, modality: str, layers: List[str]):
        owner = self.owner
        extractor = extractor_for(owner, modality, stimuli)
        if extractor is not None:
            return extractor(stimuli, layers=layers)
        modality = canonical_modality(modality)
        raise NotImplementedError(
            f"Model '{owner.identifier}' declares {modality!r} input but has no "
            'activation extractor for it. Pass extractor={modality: extractor} to '
            'record internal activity, or use a behavioral task for an output-only model.')

    def process_multi_modality(self, stimuli, detected: Set[str],
                               layers: List[str]):
        """Invoke every detected-and-supported wrapper and concat assemblies."""
        import numpy as np
        import xarray as xr

        owner = self.owner
        fused = fused_for(owner, detected)
        if fused is not None:
            # One fused extractor takes every input type: one forward pass.
            assembly = fused(stimuli, layers=layers)
            if owner._recorder.is_multi_region:
                assembly = owner._tag_neuroids_with_regions(assembly)
            return assembly
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
                    and owner._recorder.recording_regions):
                continue
            sub = (owner._process_composite_regions(stimuli, modality)
                   if owner._recorder.composite_recording else
                   owner._extract_for_modality(stimuli, modality, modality_layers))
            if not hasattr(sub, 'dims') or 'neuroid' not in sub.dims:
                raise TypeError(
                    f"multi_modality dispatch requires every extractor "
                    f"to return an xarray-shaped assembly with a 'neuroid' "
                    f"dim; modality '{modality}' returned "
                    f"{type(sub).__name__}. Pass a layer-aware extractor "
                    f"(TextWrapper / VLMVisionWrapper) via "
                    f"extractor={{modality: extractor}}."
                )
            if owner._recorder.is_multi_region and not owner._recorder.composite_recording:
                sub = owner._tag_neuroids_with_regions(sub, modality)
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
    flattened, native_times = _normalize_time_indexes(flattened)
    combined = xr.concat(flattened, dim='neuroid')
    if native_times:
        coordinates = [name for name, coord in combined.coords.items()
                       if name != 'neuroid' and coord.dims == ('neuroid',)]
        combined = combined.set_index(neuroid=coordinates)
        combined.attrs = {**combined.attrs, '_umi_native_time_indexes': native_times}
    presentation_coords = [
        name for name, coord in combined.coords.items()
        if name != 'presentation' and coord.dims == ('presentation',)
    ]
    if presentation_coords:
        combined = combined.set_index(presentation=presentation_coords)
    return combined


def _normalize_time_indexes(assemblies):
    """Join incompatible time schemas by physical bounds, never by sample number.

    Matching schemas retain the historical concat path. Source-only time metadata
    becomes a per-neuroid coordinate while the combined array uses the union of
    real intervals. The session driver restores each channel's native time axis.
    """
    import numpy as np

    timed = [assembly for assembly in assemblies if 'time_bin' in assembly.dims]
    schemas = {tuple(assembly.indexes['time_bin'].names) if 'time_bin' in assembly.indexes else ()
               for assembly in timed}
    if len(schemas) <= 1:
        return assemblies, {}
    bounds = ('time_bin_start_ms', 'time_bin_end_ms')
    normalized, native = [], {}
    for assembly in assemblies:
        if 'time_bin' not in assembly.dims:
            normalized.append(assembly)
            continue
        names = list(assembly.indexes['time_bin'].names) if 'time_bin' in assembly.indexes else []
        flat = assembly.reset_index('time_bin') if 'time_bin' in assembly.indexes else assembly
        if any(name not in flat.coords or flat[name].dims != ('time_bin',) for name in bounds):
            raise ValueError('Different modality time indexes require explicit time_bin_start_ms '
                             'and time_bin_end_ms coordinates; positional alignment is not valid.')
        if any(not np.isfinite(flat[name].values).all() for name in bounds):
            raise ValueError('Modality time-bin bounds must be finite for temporal alignment.')
        modalities = set(map(str, assembly['modality'].values))
        if len(modalities) != 1:
            raise ValueError('Each temporal block must identify one source modality.')
        modality = next(iter(modalities))
        one_dimensional = [name for name, coord in flat.coords.items()
                           if name != 'time_bin' and coord.dims == ('time_bin',)]
        native[modality] = {'index_names': names, 'one_dimensional': one_dimensional,
                            'dtypes': {name: str(flat[name].dtype) for name in one_dimensional},
                            'coordinates': [name for name, coord in flat.coords.items()
                                            if name != 'time_bin' and 'time_bin' in coord.dims]}
        shape = (flat.sizes['time_bin'], flat.sizes['neuroid'])
        extras = {name: (('time_bin', 'neuroid'), np.broadcast_to(flat[name].values[:, None], shape))
                  for name in one_dimensional if name not in bounds}
        extras['source_time_bin_index'] = (('time_bin', 'neuroid'),
            np.broadcast_to(np.arange(shape[0])[:, None], shape))
        flat = flat.assign_coords(extras).set_index(time_bin=list(bounds))
        if 'neuroid' in flat.indexes:
            flat = flat.reset_index('neuroid')
        normalized.append(flat)
    # Older supported xarray versions require every concat coordinate to exist
    # in every block. Missing source metadata is explicitly absent, not copied.
    coordinate_dims = {name: coord.dims for block in normalized
                       for name, coord in block.coords.items() if 'neuroid' in coord.dims}
    for index, block in enumerate(normalized):
        missing = {name: (dims, np.full(tuple(block.sizes[dim] for dim in dims), np.nan))
                   for name, dims in coordinate_dims.items() if name not in block.coords}
        normalized[index] = block.assign_coords(missing)
    return normalized, native


def _restore_native_time_output(assembly):
    """Remove only union-padding bins; keep genuine NaNs and source metadata."""
    import numpy as np

    native = assembly.attrs.get('_umi_native_time_indexes')
    if not native or 'source_time_bin_index' not in assembly.coords:
        return assembly
    modalities = set(map(str, assembly['modality'].values))
    if len(modalities) != 1:
        return assembly
    metadata = native.get(next(iter(modalities)))
    if metadata is None:
        return assembly
    positions = assembly['source_time_bin_index'].values
    present = np.flatnonzero(np.any(np.isfinite(positions), axis=1))
    order = np.argsort(np.nanmin(positions[present], axis=1), kind='stable')
    restored = assembly.isel(time_bin=present[order]).reset_index('time_bin')
    drop = [name for name, coord in restored.coords.items()
            if 'time_bin' in coord.dims and name not in metadata['coordinates']]
    if drop:
        restored = restored.drop_vars(drop)
    for name in metadata['one_dimensional']:
        coord = restored[name]
        values = coord.isel(neuroid=0).values if coord.dims == ('time_bin', 'neuroid') else coord.values
        restored = restored.assign_coords({name: ('time_bin', values.astype(metadata['dtypes'][name]))})
    if metadata['index_names']:
        restored = restored.set_index(time_bin=metadata['index_names'])
    restored.attrs = dict(restored.attrs)
    restored.attrs.pop('_umi_native_time_indexes', None)
    return restored


def _reset_presentation_index(assembly):
    """Expose presentation MultiIndex levels as plain coords for concat."""
    if 'presentation' not in getattr(assembly, 'indexes', {}):
        return assembly
    index = assembly.indexes['presentation']
    if index.__class__.__name__ != 'MultiIndex':
        return assembly
    return assembly.reset_index('presentation')


# Keep existing pickle references and public type names stable.
for _definition in (
    _activation_cache_disabled,
    register_input_handler,
    normalize_extractors,
    InputDispatcher,
    _concat_modalities_along_neuroid,
    _normalize_time_indexes,
    _restore_native_time_output,
    _reset_presentation_index,
):
    _definition.__module__ = 'brainscore_core.dispatch'
del _definition
