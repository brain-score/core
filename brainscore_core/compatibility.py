"""Pre-flight matching from declared input and output channels.

A partial match runs with the inputs the subject can consume. An incompatible
pair raises before compute and must be reported as N/A, never as a zero score.
"""

import warnings
from dataclasses import dataclass
from typing import Optional, Set

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


@dataclass(frozen=True)
class CompatibilityMatch:
    """Inputs a compatible pair consumes and leaves unused."""

    consumed_inputs: frozenset[str]
    dropped_inputs: frozenset[str]

    @property
    def kind(self) -> str:
        return "partial" if self.dropped_inputs else "full"

    def stamp(self, score) -> None:
        """Keep the comparison's input basis with its score."""
        score.attrs["compatibility"] = self.kind
        score.attrs["consumed_inputs"] = sorted(self.consumed_inputs)
        score.attrs["dropped_inputs"] = sorted(self.dropped_inputs)
        score.attrs["modalities_used"] = sorted({
            parse_channel(channel)[0] for channel in self.consumed_inputs
        })


def ensure_legacy_benchmark_modalities(benchmark, modalities):
    """Supply a domain default only when a legacy benchmark has no input contract."""
    if (getattr(benchmark, 'required_input_channels', None) is None
            and not getattr(benchmark, 'required_modalities', None)
            and not getattr(benchmark, 'accepted_modalities', None)):
        benchmark.required_modalities = set(modalities)
    return benchmark


def check_channel_compatibility(subject: Subject, benchmark) -> CompatibilityMatch:
    """Return the input match, or reject the pair before evaluation."""
    subject_in = set(subject.in_channels)
    subject_out = set(subject.out_channels)
    subject_required = set(subject.required_channels)
    bench_required = benchmark_required_input_channels(benchmark)
    bench_accepted = benchmark_accepted_input_channels(benchmark)
    bench_requested = benchmark_requested_output_channels(benchmark)

    validate_channels(
        subject_in, io_catalog.INPUT, f"subject '{subject.identifier}' in_channels")
    validate_channels(
        subject_out, io_catalog.OUTPUT, f"subject '{subject.identifier}' out_channels")
    validate_channels(
        subject_required, io_catalog.INPUT,
        f"subject '{subject.identifier}' required_channels")
    validate_channels(
        bench_required, io_catalog.INPUT,
        f"benchmark '{_benchmark_identifier(benchmark)}' required_input_channels")
    validate_channels(
        bench_accepted, io_catalog.INPUT,
        f"benchmark '{_benchmark_identifier(benchmark)}' accepted_input_channels")
    validate_channels(
        bench_requested, io_catalog.OUTPUT,
        f"benchmark '{_benchmark_identifier(benchmark)}' requested_output_channels")

    provided = bench_required | bench_accepted
    consumed = provided & subject_in
    if not consumed:
        raise CompatibilityError(
            f"Subject '{subject.identifier}' has no overlapping input channels "
            f"with benchmark '{_benchmark_identifier(benchmark)}': "
            f"{_format_channels(provided)}."
        )

    missing_outputs = bench_requested - subject_out
    if missing_outputs:
        raise CompatibilityError(
            f"Subject '{subject.identifier}' cannot emit requested output "
            f"channels for benchmark '{_benchmark_identifier(benchmark)}': "
            f"{_format_channels(missing_outputs)}. Subject out_channels: "
            f"{_format_channels(subject_out)}."
        )

    unmet_requirements = subject_required - (bench_required | bench_accepted)
    if unmet_requirements:
        raise CompatibilityError(
            f"Subject '{subject.identifier}' hard-requires input channels "
            f"{_format_channels(unmet_requirements)} but benchmark "
            f"'{_benchmark_identifier(benchmark)}' provides "
            f"{_format_channels(bench_required | bench_accepted)}."
        )

    return CompatibilityMatch(frozenset(consumed), frozenset(provided - consumed))


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
    """Match declarations identically for every subject implementation."""
    if getattr(benchmark, 'available_modalities', None):
        warnings.warn(
            'available_modalities is deprecated; declare all benchmark inputs '
            'with required_input_channels.', DeprecationWarning, stacklevel=2)
    return check_channel_compatibility(model, benchmark)


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
