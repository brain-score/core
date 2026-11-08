"""Record requested neural channels, each exactly as it would be recorded alone.

A session may request regions that different extractors record (a video
region and an audio region), or several regions of one extractor. Outputs of
different extractors have their own time axes and coordinates, so they are
never merged here: every channel runs the standalone path, and its payload
equals the payload of that channel requested alone. Channels run in the
model's usual source order.
"""

from typing import Dict, List, Sequence, Tuple

from ..io_catalog import canonical_modality


def run_neural_channels(model, stimuli, channel_regions: Sequence[Tuple[str, str]],
                        time_bins=None, multi_modality: bool = False) -> Dict[str, object]:
    """Return ``{channel: payload}`` for the requested ``neural:<region>`` channels."""
    regions = [region for _, region in channel_regions]
    _refuse_duplicates([channel for channel, _ in channel_regions])
    _start(model, regions[0] if len(regions) == 1 else regions, time_bins)
    state = model._recorder.snapshot()   # the combined recording, as before
    try:
        payloads = {}
        for region in _plan(model, stimuli, regions):
            _start(model, region, time_bins)
            output = _process(model, stimuli, multi_modality)
            check_layer_provenance(model, output, region)
            payloads[region] = output
        return {channel: payloads[region] for channel, region in channel_regions}
    finally:
        model._recorder.restore(state)


def _refuse_duplicates(channels: List[str]) -> None:
    repeated = sorted({channel for channel in channels if channels.count(channel) > 1})
    if repeated:
        raise ValueError(
            f'Duplicate output channels requested: {repeated}. Request each '
            'neural:<region> channel once.')


def _plan(model, stimuli, regions: List[str]) -> List[str]:
    """The order in which ``regions`` run. Nothing runs here.

    Every input is checked first. Regions run in the model's source order
    (``MODALITY_PRIORITY`` of the input each reads, then request order), the
    order in which a combined recording would run its extractors.
    """
    detected = model._detect_modalities(stimuli)
    routing = model._region_modality_map
    modality_of = {region: canonical_modality(routing[region])
                   for region in regions if region in routing}
    missing = set(modality_of.values()) - set(detected)
    if missing:
        raise ValueError(
            f'Recording targets require missing input modalities: {sorted(missing)}.')

    priority = list(model.MODALITY_PRIORITY)
    rank = lambda modality: (priority.index(modality) if modality in priority
                             else len(priority))

    def source_rank(region):
        modality = modality_of.get(region)
        return rank(modality) if modality is not None else min(
            (rank(m) for m in detected), default=len(priority))

    return sorted(regions, key=lambda r: (source_rank(r), regions.index(r)))


def check_layer_provenance(model, output, region: str) -> None:
    """Every neuroid names a layer the region records, and every required layer has one."""
    import numpy as np

    allowed, required = _region_layers(model, region)
    if not hasattr(output, 'dims') or 'neuroid' not in output.dims:
        raise ValueError(f"Neural output for region {region!r} has no 'neuroid' dimension.")
    for dim, index in output.indexes.items():
        if dim != 'neuroid' and 'layer' in (getattr(index, 'names', None) or ()):
            raise ValueError(
                f"Neural output for region {region!r} has a 'layer' coordinate on "
                f"{dim!r}; expected it on 'neuroid'.")
    if not output.sizes['neuroid']:
        raise ValueError(f'Neural output for region {region!r} contains no neuroids.')
    try:
        layer = output['layer']
    except KeyError:
        if len(allowed) == 1:
            return      # one layer: attribution is unambiguous
        raise ValueError(
            f"Neural output for region {region!r} has no per-neuroid 'layer' "
            'coordinate; attribution by concatenation order is not allowed.')
    if layer.dims != ('neuroid',):
        raise ValueError(
            f"Neural output for region {region!r} has a 'layer' coordinate with dims "
            f"{layer.dims}; expected ('neuroid',).")
    labels = set(np.asarray(layer.values).astype(str))
    unknown = labels - allowed
    if unknown:
        raise ValueError(
            f'Neural output for region {region!r} contains neuroids from layers '
            f'{sorted(unknown)}, which it does not record (expected {sorted(allowed)}).')
    uncovered = required - labels
    if uncovered:
        raise ValueError(
            f'Neural output for region {region!r} contains no neuroids for the '
            f'requested layers {sorted(uncovered)}.')


def _region_layers(model, region: str):
    """``(allowed, required)`` layers of a requested address.

    ``required_layers`` is the one rule for which layers must contribute. The
    virtual ``'all'`` covers every mapped region with the same rule.
    """
    from .recording import required_layers, selector_layers
    from ..selection import LayerSelector
    selectors = model._region_layer_selectors
    if region == 'all' and region not in selectors:
        chosen = list(selectors.values())
    else:
        chosen = [selectors.get(region, LayerSelector(region))]
    allowed = {layer for selector in chosen for layer in selector_layers(selector)}
    required = {layer for selector in chosen for layer in required_layers(selector)}
    return allowed, required


def _start(model, target, time_bins) -> None:
    if time_bins is None:
        model.start_recording(target)
    else:
        model.start_recording(target, time_bins=time_bins)


def _process(model, stimuli, multi_modality: bool):
    if multi_modality:
        return model.process(stimuli, multi_modality=True)
    return model.process(stimuli)
