"""Score metadata helpers for UMI v2.0 provenance fields.

Public names (``__all__``): the score stamp, the output channels for a score, and the
run-measure helpers. ``note_session`` is public on purpose: a subject that chooses its own
execution plan (unified ``BrainScoreModel``) calls it so the run record names that plan.
"""

from typing import Iterable, Optional, Set

from brainscore_core.compatibility import _declares_channels, benchmark_requested_output_channels
from brainscore_core._evaluation.measures import (  # noqa: F401  (re-exported)
    MeasureRecord,
    note_session,
    record_measures,
)
from brainscore_core._evaluation import measures as _measures
from brainscore_core.environment_check import _score_environment

__all__ = ['MeasureRecord', 'note_session', 'record_measures',
           'requested_output_channels_for_score', 'stamp_score_metadata']


def stamp_score_metadata(
    score,
    subject,
    requested_channels: Iterable[str],
    entry_point: Optional[str],
    measures: Optional[MeasureRecord] = None,
    *,
    benchmark=None,
    benchmark_identifier: Optional[str] = None,
):
    """Add channel and response provenance attrs to a returned score.

    The three ``score()`` functions call it after the run with the same steps.
    With ``benchmark_identifier`` (the identifier the caller asked for),
    ``attrs['benchmark_identifier']`` is that identifier, or the one the
    benchmark wrote itself when it is the benchmark's own identifier (a
    benchmark can name a qualified identity); any other value raises
    ``ValueError``. With ``benchmark``, ``attrs['trial_memory']`` is the
    benchmark's ``trial_memory`` (``'none'`` when it declares none).

    ``attrs['response']`` copies the response record of every behavioral answer
    the run's sessions accepted (``record_measures()``): a tuple of records,
    one per distinct record, sorted by format; empty when no behavioral answer
    was recorded (``brainscore_core.response_formats.response_record``).
    ``measures`` also gives the execution configuration of the run's sessions:
    ``attrs['delivery']`` (``'listed'``, ``'stream'``, both as a sorted tuple,
    or None when no session answered) and ``attrs['regions_per_pass']``
    (the plan the subject used, or None for a subject that does not read it).
    ``attrs['software']`` gives the versions of python, the libraries that can
    change numbers, the four Brain-Score packages, ``environment_fingerprint``,
    ``accelerator`` and ``math_threads`` (the threads of the math libraries);
    ``attrs['code_commits']``
    the commit of each imported Brain-Score package's checkout.
    ``attrs['provenance']`` copies the provenance entries the run's sessions
    noted (``note_session(..., provenance=)``; rule 6.3), sorted by content; a
    subject that noted none (an old model through an adapter, a custom subject)
    gets one entry whose facts are ``'not stated'``.

    ``attrs['requested_outputs']`` names the outputs the benchmark requested (not the
    subject's declared ``out_channels``), ``attrs['subject_identifier']`` the scored
    subject and ``attrs['entry_point']`` the ``score()`` that made the score
    (``'brainscore'``, ``'brainscore_vision'`` or ``'brainscore_language'``).
    """
    if benchmark_identifier is not None:
        recorded = score.attrs.get("benchmark_identifier", benchmark_identifier)
        own = _identifier_of(benchmark) or benchmark_identifier
        if not isinstance(recorded, str) or recorded not in (benchmark_identifier, own):
            raise ValueError(
                f"Score benchmark_identifier {recorded!r} must match the requested identifier "
                f"{benchmark_identifier!r} or the benchmark's identifier {own!r}"
            )
        score.attrs["benchmark_identifier"] = recorded
    if benchmark is not None:
        score.attrs["trial_memory"] = getattr(benchmark, "trial_memory", "none")
    score.attrs["in_channels"] = _sorted_tuple(getattr(subject, "in_channels", ()))
    score.attrs["subject_traits"] = _sorted_tuple(getattr(subject, "traits", ()))
    score.attrs["requested_outputs"] = _sorted_tuple(requested_channels)
    score.attrs["subject_identifier"] = getattr(subject, "identifier", None)
    score.attrs["entry_point"] = entry_point
    score.attrs["response"] = measures.responses if measures is not None else ()
    score.attrs["provenance"] = _measures.provenance_attrs(
        measures, score.attrs["subject_identifier"])
    if measures is not None:
        score.attrs.update(_measures.execution_attrs(measures))
    score.attrs.update(_score_environment())
    return score


def requested_output_channels_for_score(benchmark) -> Set[str]:
    """Return output channels requested by a benchmark for score metadata.

    Only a benchmark that declares no channels (a legacy benchmark) gets a guess from
    its ``parent``; a declaration is read as it is (rule 5.6).
    """
    requested = benchmark_requested_output_channels(benchmark)
    if requested or _declares_channels(benchmark):
        return requested

    parent = _normalise_parent(_optional_attr(benchmark, "parent"))
    if _is_behavior_parent(parent):
        return {"behavior"}
    if "closed_loop" in parent or "embodied" in parent or "motor" in parent:
        return {"motor"}
    if "perturb" in parent:
        return {"perturbation"}
    return set()


def _identifier_of(obj) -> Optional[str]:
    """The identifier of a benchmark (a property or a method), or None."""
    identifier = getattr(obj, "identifier", None)
    if callable(identifier):
        try:
            identifier = identifier()
        except Exception:
            identifier = None
    return identifier if isinstance(identifier, str) else None


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
