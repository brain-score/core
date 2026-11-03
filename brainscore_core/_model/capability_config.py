"""Capability configuration normalization for BrainScoreModel."""

from typing import Any, Dict, Optional, Set, Tuple


_CAPABILITY_CONFIG_KEYS = frozenset({
    'behavioral_readout_layer',
    'generation_fn',
    'action_fn',
    'state_change_fn',
})

_LEGACY_TAIL_ARGS = (
    'behavioral_readout_layer',
    'generation_fn',
    'action_fn',
    'state_change_fn',
    'required_modalities',
    'backbone_id',
    'region_modality_map',
)


def normalize_capability_config(
    capability_config: Optional[Dict[str, Any]],
    legacy_tail_args: Tuple[Any, ...],
    legacy_capability_config: Dict[str, Any],
    required_modalities: Optional[Set[str]],
    backbone_id: Optional[str],
    region_modality_map: Optional[Dict[str, str]],
):
    if len(legacy_tail_args) > len(_LEGACY_TAIL_ARGS):
        raise TypeError(
            f"BrainScoreModel() takes at most {6 + len(_LEGACY_TAIL_ARGS)} "
            f"positional arguments ({6 + len(legacy_tail_args)} given)"
        )

    legacy_values = dict(zip(_LEGACY_TAIL_ARGS, legacy_tail_args))
    for name in _LEGACY_TAIL_ARGS:
        if name in legacy_capability_config:
            if name in legacy_values:
                raise TypeError(
                    f"BrainScoreModel() got multiple values for argument "
                    f"'{name}'"
                )
            legacy_values[name] = legacy_capability_config.pop(name)
    if legacy_capability_config:
        unexpected = next(iter(legacy_capability_config))
        raise TypeError(
            f"BrainScoreModel() got an unexpected keyword argument "
            f"'{unexpected}'"
        )

    required_modalities = _consume_host_arg(
        legacy_values, 'required_modalities', required_modalities)
    backbone_id = _consume_host_arg(legacy_values, 'backbone_id', backbone_id)
    region_modality_map = _consume_host_arg(
        legacy_values, 'region_modality_map', region_modality_map)

    capability_values = dict(capability_config or {})
    for name, value in legacy_values.items():
        if name not in _CAPABILITY_CONFIG_KEYS:
            raise TypeError(
                f"BrainScoreModel() got an unexpected keyword argument "
                f"'{name}'"
            )
        if name in capability_values:
            raise TypeError(
                f"BrainScoreModel() got multiple capability config values "
                f"for '{name}'"
            )
        capability_values[name] = value

    return (
        capability_values,
        required_modalities,
        backbone_id,
        region_modality_map,
    )


def _consume_host_arg(legacy_values, name, current_value):
    if name not in legacy_values:
        return current_value
    if current_value is not None:
        raise TypeError(
            f"BrainScoreModel() got multiple values for argument '{name}'"
        )
    return legacy_values.pop(name)
