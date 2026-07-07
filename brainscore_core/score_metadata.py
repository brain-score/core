"""Score metadata helpers for UMI v2.0 provenance fields."""

from typing import Iterable, Optional, Set

from .compatibility import benchmark_requested_output_channels
from .streaming import parse_channel


def stamp_score_metadata(
    score,
    subject,
    requested_channels: Iterable[str],
    protocol: Optional[str],
    harness_id: Optional[str],
):
    """Add channel/protocol provenance attrs to a returned score."""
    score.attrs["in_channels"] = _sorted_tuple(getattr(subject, "in_channels", ()))
    score.attrs["out_channels"] = _sorted_tuple(requested_channels)
    score.attrs["protocol"] = protocol
    score.attrs["subject_id"] = getattr(subject, "identifier", None)
    score.attrs["harness_id"] = harness_id
    return score


def requested_output_channels_for_score(benchmark) -> Set[str]:
    """Return output channels requested by a benchmark for score metadata."""
    requested = benchmark_requested_output_channels(benchmark)
    if requested:
        return requested

    parent = _normalise_parent(_optional_attr(benchmark, "parent"))
    if _is_behavior_parent(parent):
        return {"behavior"}
    if "embodied" in parent or "motor" in parent:
        return {"motor"}
    if "perturb" in parent:
        return {"perturbation"}
    return set()


def infer_score_protocol(subject, benchmark, requested_channels: Iterable[str]) -> Optional[str]:
    """Infer the high-level UMI protocol path used by a score."""
    families = {_channel_family(channel) for channel in requested_channels}
    if "motor" in families:
        return "motor"
    if "perturbation" in families:
        return "perturbation"
    if "behavior" in families:
        return _behavior_protocol(subject)
    if "neural" in families:
        return "neural"

    parent = _normalise_parent(_optional_attr(benchmark, "parent"))
    if _is_behavior_parent(parent):
        return _behavior_protocol(subject)
    if "embodied" in parent or "motor" in parent:
        return "motor"
    if "perturb" in parent:
        return "perturbation"
    if "neural" in parent or "naturalistic" in parent:
        return "neural"
    return None


def _sorted_tuple(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values), key=repr))


def _channel_family(channel: str) -> str:
    try:
        family, _address = parse_channel(channel)
    except (TypeError, ValueError):
        return str(channel).split(":", 1)[0]
    return family


def _normalise_parent(parent) -> str:
    if parent is None:
        return ""
    return str(parent).lower()


def _optional_attr(obj, name: str):
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _is_behavior_parent(parent: str) -> bool:
    return "behavior" in parent or "engineering" in parent


def _behavior_protocol(subject) -> str:
    if getattr(subject, "_use_generation_for_task", False) is True:
        return "behavior:generation"
    return "behavior:readout"
