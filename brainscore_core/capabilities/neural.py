"""Neural encoding capability."""

from .base import Capability
from .registry import register_capability


class NeuralEncodingCapability(Capability):
    """Default StimulusSet -> NeuroidAssembly path."""

    identifier = 'neural-encoding'
    order = 1000

    def enabled_for(self, model) -> bool:
        return bool(model._preprocessors)

    def handles(self, model, event, **kwargs) -> bool:
        del kwargs
        try:
            event.columns
        except AttributeError:
            return False
        return True

    def process(self, model, stimuli, **kwargs):
        multi_modality = kwargs.get('multi_modality', False)
        detected = model._detect_modalities(stimuli)

        if not detected:
            raise ValueError(
                f"No recognized modality columns in stimulus set. "
                f"Columns present: {list(stimuli.columns)}. "
                f"Known column mappings: {model.COLUMN_TO_MODALITY}. "
                f"Model supports: {model.supported_modalities}."
            )

        layers = model._recorder.current_layers()
        modality = model._pick_modality(detected)

        # Guard BOTH paths: if regions ARE defined but none is active, the user
        # forgot start_recording (an empty region_layer_map is a legitimate
        # default-extraction path). Check only the modality/modalities that will
        # ACTUALLY run — the selected one for single dispatch, all detected for a
        # multi-modality fan-out — so a non-selected layer-aware tower does not
        # false-positive the selected plain-callable path.
        will_run = detected if (multi_modality and len(detected) > 1) else {modality}
        if (not layers and model._region_layer_map_dict
                and not model._composite_recording
                and any(model._supports_layer_extraction(m) for m in will_run)):
            raise ValueError(
                f"No active recording layer for model '{model.identifier}': "
                f"call start_recording(<region>) before process(), otherwise "
                f"extraction returns 0 neuroids. Known regions: "
                f"{sorted(model._region_layer_map_dict.keys())}."
            )

        if multi_modality and len(detected) > 1:
            return model._process_multi_modality(stimuli, detected, layers)

        if not multi_modality and len(detected) > 1:
            import warnings
            warnings.warn(
                f"Stimulus set contains multiple supported modalities "
                f"{sorted(detected)} for model '{model.identifier}'. "
                f"Default process(..., multi_modality=False) will use "
                f"'{modality}' via MODALITY_PRIORITY. Pass "
                f"multi_modality=True to extract all supported modalities, "
                f"or provide stimulus columns for only the modality you want.",
                UserWarning,
                stacklevel=2,
            )
        if (model._composite_recording
                and model._supports_layer_extraction(modality)):
            return model._process_composite_regions(stimuli, modality)
        assembly = model._extract_for_modality(stimuli, modality, layers)
        if model._is_multi_region and model._supports_layer_extraction(modality):
            assembly = model._tag_neuroids_with_regions(assembly)
        return assembly


register_capability(NeuralEncodingCapability())
