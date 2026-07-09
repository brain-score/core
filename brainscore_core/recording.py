"""Recording state and region/layer resolution for BrainScoreModel."""

from typing import Dict, List, Optional, Tuple, Union

from .selection import CompositeSelector


class Recorder:
    """Owns recording target state for a BrainScoreModel instance."""

    def __init__(self, owner) -> None:
        self.owner = owner
        self.recording_layer: Optional[str] = None
        self.recording_regions: List[str] = []
        self.recording_layers: List[str] = []
        self.is_multi_region: bool = False
        self.time_bins: Optional[List[Tuple[int, int]]] = None
        self.composite_recording: bool = False

    def start_recording(self,
                        recording_target: Union[str, List[str]],
                        time_bins: Optional[List[Tuple[int, int]]] = None,
                        recording_type: Optional[str] = None) -> None:
        """Configure which model layers to extract."""
        del recording_type
        owner = self.owner

        if recording_target == 'all':
            recording_target = list(owner._region_layer_map_dict.keys())
            if not recording_target:
                raise ValueError(
                    "start_recording('all') requires a non-empty "
                    "region_layer_map."
                )

        if (isinstance(recording_target, str)
                and recording_target in owner._region_layer_selectors
                and isinstance(owner._region_layer_selectors[recording_target],
                               CompositeSelector)):
            recording_target = [recording_target]

        if isinstance(recording_target, str):
            if recording_target not in owner._region_layer_map_dict:
                import warnings
                warnings.warn(
                    f"Recording target '{recording_target}' is not a region in "
                    f"region_layer_map (known regions: "
                    f"{sorted(owner._region_layer_map_dict.keys())}); treating it "
                    f"as a raw layer path. If you meant a region, check the spelling.",
                    UserWarning, stacklevel=2)
            self.recording_layer = owner._region_layer_map_dict.get(
                recording_target, recording_target
            )
            self.recording_regions = (
                [recording_target]
                if recording_target in owner._region_layer_map_dict
                else []
            )
            self.recording_layers = [self.recording_layer]
            self.is_multi_region = False
            self.composite_recording = False
        else:
            regions = list(recording_target)
            unknown = [r for r in regions
                       if r not in owner._region_layer_map_dict]
            if unknown:
                raise ValueError(
                    f"Region(s) {unknown} not in region_layer_map "
                    f"(known regions: "
                    f"{list(owner._region_layer_map_dict.keys())})"
                )
            self.recording_regions = regions
            all_layers: List[str] = []
            for r in regions:
                sel = owner._region_layer_selectors[r]
                if isinstance(sel, CompositeSelector):
                    all_layers.extend(sel.layer_paths)
                else:
                    all_layers.append(sel.layer_path)
            self.recording_layers = list(dict.fromkeys(all_layers))
            self.recording_layer = (
                self.recording_layers[0]
                if len(self.recording_layers) == 1 else None
            )
            self.composite_recording = any(
                isinstance(owner._region_layer_selectors[r], CompositeSelector)
                for r in regions
            )
            self.is_multi_region = len(regions) > 1
        self.time_bins = time_bins

    def reset(self) -> None:
        self.recording_layer = None
        self.recording_layers = []
        self.recording_regions = []
        self.is_multi_region = False

    def current_layers(self) -> List[str]:
        if self.recording_layers:
            return list(self.recording_layers)
        if self.recording_layer:
            return [self.recording_layer]
        return []

    def filter_layers_for_modality(self, layers: List[str],
                                   modality: str) -> List[str]:
        """Return the subset of ``layers`` that belong to ``modality``."""
        owner = self.owner
        if not owner._region_modality_map:
            return list(layers)
        layer_to_modality: Dict[str, str] = {}
        for region, m in owner._region_modality_map.items():
            layer_path = owner._region_layer_map_dict[region]
            layer_to_modality.setdefault(layer_path, m)
        return [
            layer for layer in layers
            if layer_to_modality.get(layer, modality) == modality
        ]

    def tag_neuroids_with_regions(self, assembly):
        """Add a 'region' coord to the neuroid axis of a multi-layer assembly."""
        import numpy as np

        owner = self.owner
        if 'layer' not in assembly.coords:
            return assembly
        layer_to_regions: Dict[str, List[str]] = {}
        for region in self.recording_regions:
            layer = owner._region_layer_map_dict[region]
            layer_to_regions.setdefault(layer, []).append(region)

        neuroid_layers = assembly['layer'].values
        neuroid_regions = np.array([
            '|'.join(layer_to_regions.get(layer, []))
            for layer in neuroid_layers
        ])
        return assembly.assign_coords(region=('neuroid', neuroid_regions))

    def process_composite_regions(self, stimuli, modality):
        """Extract and concatenate blocks for composite recording regions."""
        import numpy as np
        import xarray as xr

        owner = self.owner
        blocks = []
        for region in self.recording_regions:
            selector = owner._region_layer_selectors[region]
            layer_paths = (list(selector.layer_paths)
                           if isinstance(selector, CompositeSelector)
                           else [selector.layer_path])
            assembly = owner._extract_for_modality(stimuli, modality, layer_paths)
            block = self.gather_composite_units(assembly, selector)
            n = block.sizes['neuroid']
            block = block.assign_coords(
                region=('neuroid', np.array([region] * n)))
            blocks.append(block)
        return xr.concat(blocks, dim='neuroid')

    def gather_composite_units(self, assembly, selector):
        """Keep the units selected from a possibly multi-layer assembly."""
        import numpy as np

        if not isinstance(selector, CompositeSelector):
            return assembly
        layer_coord = assembly['layer'].values
        keep_positions: List[int] = []
        for layer_path, indices in selector.layers:
            layer_positions = np.where(layer_coord == layer_path)[0]
            if indices is None:
                keep_positions.extend(layer_positions.tolist())
            else:
                keep_positions.extend(layer_positions[list(indices)].tolist())
        return assembly.isel(neuroid=keep_positions)
