"""
Pre-flight compatibility checker for Brain-Score (Contract Rules v3, rule 5).

Compatibility is decided from declarations alone: what the benchmark sends
and asks for, and what the subject says it takes in, produces and requires.
A subject is compatible with a benchmark when all three hold:

1. every requested output is in ``subject.out_channels``;
2. at least one benchmark input is in ``subject.in_channels``;
3. every channel in ``subject.required_channels`` is among the benchmark's
   inputs.

The benchmark declares the inputs its human participants received. A subject
that takes in only some of them still runs: the streams it cannot take in
never enter it. :func:`check_compatibility` returns the match as data
(:class:`CompatibilityMatch`) so the score records which inputs were used.
An incompatible pair raises :class:`CompatibilityError` and is scored N/A.

Benchmark inputs are ``required_input_channels`` when declared, otherwise the
channels of ``required_modalities`` plus ``accepted_modalities``. Requested
outputs are ``requested_output_channels`` when declared, otherwise
``neural:<region>`` for a benchmark with a ``region``.

Benchmarks that still set ``available_modalities`` emit a
``DeprecationWarning`` directing the maintainer to register separate variants.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Set

from . import io_catalog
from .io_catalog import canonical_modality, modalities_to_input_channels
from .model_interface import Subject
from .streaming import parse_channel


class CompatibilityError(Exception):
    """Raised when a model cannot be scored on a benchmark."""
    pass


class CompatibilityWarning(UserWarning):
    """Issued when a model can run but will not use all available modalities."""
    pass


def ensure_legacy_benchmark_modalities(benchmark, modalities):
    """Supply a domain default only when a legacy benchmark has no input contract."""
    if (getattr(benchmark, 'required_input_channels', None) is None
            and not getattr(benchmark, 'required_modalities', None)
            and not getattr(benchmark, 'accepted_modalities', None)):
        benchmark.required_modalities = set(modalities)
    return benchmark


@dataclass(frozen=True)
class CompatibilityMatch:
    """How a subject matches a benchmark, returned by pre-flight.

    ``consumed_inputs`` are the benchmark inputs the subject takes in and
    ``dropped_inputs`` the ones it cannot, which never enter it. ``kind`` is
    ``'full'`` when nothing is dropped, otherwise ``'partial'``.
    """
    subject: str
    benchmark: str
    consumed_inputs: FrozenSet[str]
    dropped_inputs: FrozenSet[str]
    requested_outputs: FrozenSet[str]

    @property
    def kind(self) -> str:
        return 'partial' if self.dropped_inputs else 'full'

    def as_attrs(self) -> Dict[str, object]:
        """Plain-data form for ``score.attrs['compatibility']``."""
        return {
            'match': self.kind,
            'consumed_inputs': sorted(self.consumed_inputs),
            'dropped_inputs': sorted(self.dropped_inputs),
            'requested_outputs': sorted(self.requested_outputs),
        }


def check_channel_compatibility(subject: Subject, benchmark) -> CompatibilityMatch:
    """Check the three compatibility conditions and return the match.

    Raises :class:`CompatibilityError` before compute when a condition fails.
    Every declared channel must be registered in the catalog with the right
    direction.
    """
    subject_in = set(subject.in_channels)
    subject_out = set(subject.out_channels)
    subject_required = set(subject.required_channels)
    bench_inputs = benchmark_input_channels(benchmark)
    bench_requested = benchmark_requested_output_channels(benchmark)
    subject_id = subject.identifier
    benchmark_id = _benchmark_identifier(benchmark)

    validate_channels(subject_in, io_catalog.INPUT, f"subject '{subject_id}' in_channels")
    validate_channels(subject_out, io_catalog.OUTPUT, f"subject '{subject_id}' out_channels")
    validate_channels(subject_required, io_catalog.INPUT,
                      f"subject '{subject_id}' required_channels")
    validate_channels(bench_inputs, io_catalog.INPUT,
                      f"benchmark '{benchmark_id}' input channels")
    validate_channels(bench_requested, io_catalog.OUTPUT,
                      f"benchmark '{benchmark_id}' requested_output_channels")

    missing_outputs = bench_requested - subject_out
    if missing_outputs:
        message = (
            f"Subject '{subject_id}' cannot produce output channels requested by "
            f"benchmark '{benchmark_id}': {_format_channels(missing_outputs)}. "
            f"Subject out_channels: {_format_channels(subject_out)}."
        )
        regions = sorted(address for family, address in map(parse_channel, missing_outputs)
                         if family == 'neural' and address is not None)
        if regions:
            message += (
                f" It has no layer mapping for region '{regions[0]}'. Use the "
                f"layer-mapping explorer (brainscore_vision.tools.layer_mapping_explorer) "
                f"to find and commit a region-to-layer mapping."
            )
        raise CompatibilityError(message)

    consumed = bench_inputs & subject_in
    if bench_inputs and not consumed:
        raise CompatibilityError(
            f"Subject '{subject_id}' takes in none of the inputs benchmark "
            f"'{benchmark_id}' sends: {_format_channels(bench_inputs)}. "
            f"Subject in_channels: {_format_channels(subject_in)}."
        )

    unmet_requirements = subject_required - bench_inputs
    if unmet_requirements:
        raise CompatibilityError(
            f"Subject '{subject_id}' hard-requires input channels "
            f"{_format_channels(unmet_requirements)} but benchmark "
            f"'{benchmark_id}' sends {_format_channels(bench_inputs)}."
        )

    if getattr(benchmark, "uses_session", False):
        probe = getattr(subject, "check_session_support", None)
        if callable(probe):
            try:
                probe(bench_requested)
            except (ValueError, NotImplementedError) as error:
                raise CompatibilityError(str(error)) from error

    return CompatibilityMatch(
        subject=subject_id,
        benchmark=benchmark_id,
        consumed_inputs=frozenset(consumed),
        dropped_inputs=frozenset(bench_inputs - subject_in),
        requested_outputs=frozenset(bench_requested),
    )


def benchmark_input_channels(benchmark) -> Set[str]:
    """Every input channel the benchmark sends."""
    return benchmark_required_input_channels(benchmark) | benchmark_accepted_input_channels(benchmark)


def benchmark_required_input_channels(benchmark) -> Set[str]:
    """Return v2 required input channels, deriving from v1.5 modalities."""
    declared = getattr(benchmark, "required_input_channels", None)
    if declared is not None:
        return set(declared)
    modalities = set(getattr(benchmark, "required_modalities", set()))
    return modalities_to_input_channels(modalities)


def benchmark_accepted_input_channels(benchmark) -> Set[str]:
    """Return v1.5 any-of input channels from accepted_modalities."""
    if getattr(benchmark, "required_input_channels", None) is not None:
        return set()
    modalities = set(getattr(benchmark, "accepted_modalities", set()))
    return modalities_to_input_channels(modalities)


def benchmark_requested_output_channels(benchmark) -> Set[str]:
    """Return v2 requested output channels, with a v1.5 region fallback."""
    declared = getattr(benchmark, "requested_output_channels", None)
    if declared is not None:
        return set(declared)
    region = getattr(benchmark, "region", None)
    if region is None:
        return set()
    return {f"neural:{region}"}


def validate_channels(channels: Set[str], direction: str, owner: str, *,
                      allow_unregistered: bool = False) -> None:
    """Validate channel names and direction against the shared catalog.

    Exploratory experiments may allow unregistered families with a warning.
    Benchmark scoring requires registration. Invalid syntax or direction always
    raises CompatibilityError.
    """
    for channel in sorted(channels, key=repr):
        try:
            family, address = parse_channel(channel)
        except (TypeError, ValueError) as error:
            raise CompatibilityError(
                f"{owner} declares invalid channel '{channel}': {error}."
            ) from error

        try:
            entry = io_catalog.get(family)
        except KeyError as error:
            if allow_unregistered:
                warnings.warn(
                    f"{owner} declares unregistered channel '{channel}'. "
                    'This experiment may run, but benchmark scoring requires '
                    'a catalog channel or register_channel() registration.',
                    CompatibilityWarning, stacklevel=2)
                continue
            raise CompatibilityError(
                f"{owner} declares unknown channel '{channel}'."
            ) from error

        if entry.direction not in (direction, io_catalog.BOTH):
            raise CompatibilityError(
                f"{owner} declares channel '{channel}' as {direction}, but "
                f"the registry marks family '{family}' as {entry.direction}."
            )

        if address is not None and entry.addressing is None:
            raise CompatibilityError(
                f"{owner} declares addressed channel '{channel}', but family "
                f"'{family}' is not addressable."
            )


def _format_channels(channels: Set[str]) -> str:
    return (
        "{"
        + ", ".join(repr(channel) for channel in sorted(channels, key=repr))
        + "}"
    )


def _benchmark_identifier(benchmark) -> str:
    return getattr(benchmark, "identifier", "?")


def check_compatibility(model: Subject, benchmark) -> CompatibilityMatch:
    """Fast pre-flight check; returns the match or raises CompatibilityError.

    One rule for every subject (see the module docstring). Emits a
    :class:`DeprecationWarning` if the benchmark still sets the deprecated
    ``available_modalities`` field, before any compatibility error.
    """
    bench_required: Set[str] = {canonical_modality(m)
                                for m in getattr(benchmark, 'required_modalities', set())}
    bench_available_attr = getattr(benchmark, 'available_modalities', None)
    if bench_available_attr is not None:
        bench_available: Set[str] = {canonical_modality(m) for m in bench_available_attr}
        # Only warn if it adds something beyond required (otherwise it is a
        # redundant declaration, not a multi-input claim).
        if bench_available - bench_required:
            warnings.warn(
                f"Benchmark '{_benchmark_identifier(benchmark)}' sets "
                f"`available_modalities={bench_available}` — this field is "
                f"deprecated as of the April 30, 2026 design decision. "
                f"Benchmarks now declare exactly one input format via "
                f"`required_modalities`. Multiple input formats should be "
                f"registered as separate benchmark variants. See "
                f"meeting-unified-interface-2026-04-30 for context.",
                DeprecationWarning,
                stacklevel=2,
            )

    match = check_channel_compatibility(model, benchmark)
    # Warn-only documentation conformance; never changes the result.
    check_io_catalog(model, benchmark)
    return match


def check_io_catalog(model: Subject, benchmark) -> None:
    """Optional documentation-conformance check against the Input/Output Catalog.

    Warn-only: emits a :class:`CompatibilityWarning` for any input the benchmark
    declares, or any neural output the benchmark targets, that is not documented
    in the catalog. It never raises and does not change the pass/fail result of
    :func:`check_compatibility`; it only makes undocumented channels visible at
    pre-flight rather than discovered later.
    """
    bench_required: Set[str] = set(getattr(benchmark, 'required_modalities', set()))
    for modality in bench_required:
        if not io_catalog.has(modality):
            warnings.warn(
                f"Benchmark '{getattr(benchmark, 'identifier', '?')}' declares input "
                f"modality '{modality}' which is not in the Input/Output Catalog. "
                f"Documented inputs: {sorted(e.name for e in io_catalog.inputs())}.",
                CompatibilityWarning,
                stacklevel=2,
            )

    region: Optional[str] = getattr(benchmark, 'region', None)
    if region is not None and not io_catalog.has(f'neural:{region}'):
        warnings.warn(
            f"Benchmark '{getattr(benchmark, 'identifier', '?')}' targets neural "
            f"region '{region}' but 'neural' is not documented in the Input/Output "
            f"Catalog.",
            CompatibilityWarning,
            stacklevel=2,
        )
