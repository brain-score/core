"""Score metadata helpers for UMI v2.0 provenance fields."""

from typing import Iterable, Optional, Set

from brainscore_core.compatibility import benchmark_requested_output_channels
from brainscore_core._evaluation.measures import (  # noqa: F401  (re-exported)
    MeasureRecord,
    note_session,
    record_measures,
)
from brainscore_core._evaluation import measures as _measures
from brainscore_core.environment_check import _score_environment


def stamp_score_metadata(
    score,
    subject,
    requested_channels: Iterable[str],
    harness_id: Optional[str],
    measures: Optional[MeasureRecord] = None,
):
    """Add channel and response provenance attrs to a returned score.

    ``attrs['response']`` copies the response record of every behavioral answer
    the run's sessions accepted (``record_measures()``): a tuple of records,
    one per distinct record, sorted by format; empty when no behavioral answer
    was recorded (``brainscore_core.response_formats.response_record``).
    ``measures`` also gives the execution configuration of the run's sessions:
    ``attrs['delivery']`` (``'listed'``, ``'stream'``, both as a sorted tuple,
    or None when no session answered) and ``attrs['regions_per_pass']``
    (the plan the subject used, or None for a subject that does not read it).
    ``attrs['software']`` gives the versions of python, the libraries that can
    change numbers and the four Brain-Score packages; ``attrs['code_commits']``
    the commit of each imported Brain-Score package's checkout.
    """
    score.attrs["in_channels"] = _sorted_tuple(getattr(subject, "in_channels", ()))
    score.attrs["subject_traits"] = _sorted_tuple(getattr(subject, "traits", ()))
    score.attrs["out_channels"] = _sorted_tuple(requested_channels)
    score.attrs["subject_id"] = getattr(subject, "identifier", None)
    score.attrs["harness_id"] = harness_id
    score.attrs["response"] = measures.responses if measures is not None else ()
    if measures is not None:
        score.attrs.update(_measures.execution_attrs(measures))
    score.attrs.update(_score_environment())
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


def _sorted_tuple(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values), key=repr))


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
