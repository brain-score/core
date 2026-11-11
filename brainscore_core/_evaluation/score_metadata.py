"""Score metadata helpers for UMI v2.0 provenance fields."""

from typing import Iterable, Optional, Set

from brainscore_core._evaluation.compatibility import benchmark_requested_output_channels
from brainscore_core._evaluation.measures import (  # noqa: F401  (re-exported)
    MeasureRecord,
    note_measure,
    record_measures,
)
from brainscore_core.streaming import parse_channel


def stamp_score_metadata(
    score,
    subject,
    requested_channels: Iterable[str],
    protocol: Optional[str],
    harness_id: Optional[str],
    measures: Optional[MeasureRecord] = None,
):
    """Add channel/protocol provenance attrs to a returned score.

    ``measures`` is the record ``record_measures()`` kept during the run. When
    it holds behavioral measures, ``attrs['measure']`` says how they were
    measured: ``'reported'``, ``'implied'`` or both as a sorted tuple.
    """
    score.attrs["in_channels"] = _sorted_tuple(getattr(subject, "in_channels", ()))
    score.attrs["subject_traits"] = _sorted_tuple(getattr(subject, "traits", ()))
    score.attrs["out_channels"] = _sorted_tuple(requested_channels)
    score.attrs["protocol"] = protocol
    score.attrs["subject_id"] = getattr(subject, "identifier", None)
    score.attrs["harness_id"] = harness_id
    kinds = sorted(measures.kinds) if measures is not None else []
    if kinds:
        score.attrs["measure"] = kinds[0] if len(kinds) == 1 else tuple(kinds)
    return score


def requested_output_channels_for_score(benchmark) -> Set[str]:
    """Return output channels requested by a benchmark for score metadata."""
    requested = benchmark_requested_output_channels(benchmark)
    if requested:
        return requested

    parent = _normalise_parent(_optional_attr(benchmark, "parent"))
    if _is_behavior_parent(parent):
        return {"behavior"}
    if "closed_loop" in parent or "embodied" in parent or "motor" in parent:
        return {"motor"}
    if "perturb" in parent:
        return {"perturbation"}
    return set()


def infer_score_protocol(subject, benchmark, requested_channels: Iterable[str],
                         measures: Optional[MeasureRecord] = None) -> Optional[str]:
    """Infer the high-level UMI protocol path used by a score.

    A behavioral protocol comes from ``measures``, the measures of the
    responses the run returned (``record_measures()``), never from the
    subject's state. ``subject`` is kept for callers and not read.
    """
    del subject
    families = {_channel_family(channel) for channel in requested_channels}
    if "motor" in families:
        return "motor"
    if "perturbation" in families:
        return "perturbation"
    if "behavior" in families:
        return _behavior_protocol(measures)
    if "neural" in families:
        return "neural"

    parent = _normalise_parent(_optional_attr(benchmark, "parent"))
    if _is_behavior_parent(parent):
        return _behavior_protocol(measures)
    if "closed_loop" in parent or "embodied" in parent or "motor" in parent:
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


def _behavior_protocol(measures: Optional[MeasureRecord]) -> str:
    """The behavioral path, from the measures the responses stated.

    A reported answer (a parsed reply) is generation, even when implied
    probabilities came with it. Implied answers name their source: a fitted
    readout is ``behavior:readout``; label likelihoods from the model itself
    are ``behavior:likelihood``, not a readout; an implied measure with any
    other source is ``behavior:implied``.
    """
    if measures is None or not measures.measures:
        # ponytail: nothing recorded (a benchmark that bypasses the core
        # helpers, or a caller without record_measures) keeps the old default
        # stamp so no existing stamp changes; drop it once every path records.
        return "behavior:readout"
    if "reported" in measures.kinds:
        return "behavior:generation"
    if "readout" in measures.sources:
        return "behavior:readout"
    if "likelihood" in measures.sources:
        return "behavior:likelihood"
    return "behavior:implied"


# Keep existing pickle references and public type names stable.
for _definition in (
    stamp_score_metadata,
    requested_output_channels_for_score,
    infer_score_protocol,
    _sorted_tuple,
    _channel_family,
    _normalise_parent,
    _optional_attr,
    _is_behavior_parent,
    _behavior_protocol,
):
    _definition.__module__ = 'brainscore_core.score_metadata'
del _definition
