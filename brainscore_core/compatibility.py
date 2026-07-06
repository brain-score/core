"""
Pre-flight compatibility checker for Brain-Score.

Single-direction modality validation (post April 30, 2026 design decision):

- Benchmark declares ``required_modalities`` — the modalities its stimuli
  provide. Each leaderboard benchmark declares **exactly one** input format;
  multiple input formats become separate registered variants.
- Model declares ``available_modalities`` (everything it can consume) and
  optionally ``required_modalities`` (hard requirement — model cannot run
  without these in the stimulus set, e.g. a locked-fusion model).

A pairing is valid when:

- ``benchmark.required ⊆ model.available`` — model can process ALL of the
  benchmark's required modalities (all-of).
- ``benchmark.accepted ∩ model.available ≠ ∅`` (if ``accepted_modalities`` is
  set) — the model provides AT LEAST ONE accepted input format (any-of). This
  is for benchmarks that adapt to the candidate, e.g. a naturalistic video
  benchmark that runs native-video models and falls back to frame-aggregation
  for still-image (vision) models, so it accepts ``{video, vision}``.
- ``benchmark.region ∈ model.region_layer_map`` (if the benchmark targets a
  neural region).

The earlier symmetric check ``model.required ⊆ benchmark.required ∪
benchmark.available`` was dropped: ``benchmark.available_modalities`` is
deprecated. A locked-fusion model whose required modalities aren't present
in the benchmark's single input format is caught by check #1 (its required
set is a superset of the benchmark's required set, so the benchmark's
required set isn't a subset of the model's available set unless it
exactly matches).

For models with required_modalities that don't include the benchmark's
required modality, the symmetric check is still useful — e.g. a model that
hard-requires text + vision but the benchmark only provides vision. We keep
that case via a tightened check on model_required: the model's required set
must be a subset of the benchmark's required set (the benchmark's only
input format).

Benchmarks that still set ``available_modalities`` emit a
``DeprecationWarning`` directing the maintainer to register separate variants.
"""

import warnings
from typing import Optional, Set

from . import io_catalog
from .io_catalog import modalities_to_input_channels
from .model_interface import Subject
from .streaming import parse_channel


class CompatibilityError(Exception):
    """Raised when a model cannot be scored on a benchmark."""
    pass


class CompatibilityWarning(UserWarning):
    """Issued when a model can run but will not use all available modalities."""
    pass


def check_channel_compatibility(subject: Subject, benchmark) -> None:
    """Check the UMI v2.0 channel compatibility contract.

    Raises :class:`CompatibilityError` before compute when any of the three
    channel-set conditions fail:

    1. ``benchmark.required_input_channels <= subject.in_channels``
    2. ``benchmark.requested_output_channels <= subject.out_channels``
    3. ``subject.required_channels <= benchmark.required_input_channels``
    """
    subject_in = set(subject.in_channels)
    subject_out = set(subject.out_channels)
    subject_required = set(subject.required_channels)
    bench_required = benchmark_required_input_channels(benchmark)
    bench_requested = benchmark_requested_output_channels(benchmark)

    _validate_channel_set(
        subject_in, io_catalog.INPUT, f"subject '{subject.identifier}' in_channels")
    _validate_channel_set(
        subject_out, io_catalog.OUTPUT, f"subject '{subject.identifier}' out_channels")
    _validate_channel_set(
        subject_required, io_catalog.INPUT,
        f"subject '{subject.identifier}' required_channels")
    _validate_channel_set(
        bench_required, io_catalog.INPUT,
        f"benchmark '{_benchmark_identifier(benchmark)}' required_input_channels")
    _validate_channel_set(
        bench_requested, io_catalog.OUTPUT,
        f"benchmark '{_benchmark_identifier(benchmark)}' requested_output_channels")

    missing_inputs = bench_required - subject_in
    if missing_inputs:
        raise CompatibilityError(
            f"Subject '{subject.identifier}' is missing required input "
            f"channels for benchmark '{_benchmark_identifier(benchmark)}': "
            f"{_format_channels(missing_inputs)}. Subject in_channels: "
            f"{_format_channels(subject_in)}."
        )

    missing_outputs = bench_requested - subject_out
    if missing_outputs:
        raise CompatibilityError(
            f"Subject '{subject.identifier}' cannot emit requested output "
            f"channels for benchmark '{_benchmark_identifier(benchmark)}': "
            f"{_format_channels(missing_outputs)}. Subject out_channels: "
            f"{_format_channels(subject_out)}."
        )

    unmet_requirements = subject_required - bench_required
    if unmet_requirements:
        raise CompatibilityError(
            f"Subject '{subject.identifier}' hard-requires input channels "
            f"{_format_channels(unmet_requirements)} but benchmark "
            f"'{_benchmark_identifier(benchmark)}' provides "
            f"{_format_channels(bench_required)}."
        )


def benchmark_required_input_channels(benchmark) -> Set[str]:
    """Return v2 required input channels, deriving from v1.5 modalities."""
    declared = getattr(benchmark, "required_input_channels", None)
    if declared is not None:
        return set(declared)
    modalities = set(getattr(benchmark, "required_modalities", set()))
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


def _validate_channel_set(channels: Set[str], direction: str, owner: str) -> None:
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


def check_compatibility(model: Subject, benchmark) -> None:
    """Fast pre-flight check (see module docstring for the full contract).

    Raises :class:`CompatibilityError` for hard mismatches.
    Emits :class:`DeprecationWarning` if the benchmark still sets the
    deprecated ``available_modalities`` field.
    """
    model_available: Set[str] = set(model.available_modalities)
    model_required: Set[str] = set(model.required_modalities)
    bench_required: Set[str] = set(getattr(benchmark, 'required_modalities', set()))

    # Deprecation warning — must fire before any short-circuit error so
    # benchmark maintainers see it even when the pairing is incompatible.
    bench_available_attr = getattr(benchmark, 'available_modalities', None)
    if bench_available_attr is not None:
        bench_available: Set[str] = set(bench_available_attr)
        # Only warn if it actually adds something beyond required (otherwise
        # it's a redundant declaration not a multi-input claim).
        if bench_available - bench_required:
            warnings.warn(
                f"Benchmark '{benchmark.identifier}' sets "
                f"`available_modalities={bench_available}` — this field is "
                f"deprecated as of the April 30, 2026 design decision. "
                f"Benchmarks now declare exactly one input format via "
                f"`required_modalities`. Multiple input formats should be "
                f"registered as separate benchmark variants. See "
                f"meeting-unified-interface-2026-04-30 for context.",
                DeprecationWarning,
                stacklevel=2,
            )

    # Check 1: Hard gate — benchmark's required modalities must ALL be
    # consumable by the model.
    missing_required = bench_required - model_available
    if missing_required:
        raise CompatibilityError(
            f"Model '{model.identifier}' does not support modalities required by "
            f"benchmark '{benchmark.identifier}': {missing_required}. "
            f"Model available modalities: {model_available}."
        )

    # Check 1b: Any-of gate — some benchmarks accept ANY ONE of several input
    # formats (e.g. a naturalistic benchmark that runs native-video models AND
    # falls back to frame-aggregation for still-image models). Declared as
    # `accepted_modalities`; the model must provide at least one. Distinct from
    # `required_modalities` (all-of); empty/absent means no any-of constraint.
    bench_accepted: Set[str] = set(getattr(benchmark, 'accepted_modalities', set()))
    if bench_accepted and not (bench_accepted & model_available):
        raise CompatibilityError(
            f"Model '{model.identifier}' provides none of the input formats "
            f"benchmark '{benchmark.identifier}' accepts; needs at least one of "
            f"{bench_accepted}, model available: {model_available}."
        )

    # Check 2: Hard gate — model's required modalities must be a subset of the
    # modalities the benchmark can provide (its required set plus any any-of
    # accepted formats). A locked-fusion model that needs modalities the
    # benchmark never provides cannot run.
    model_missing = model_required - (bench_required | bench_accepted)
    if model_missing:
        raise CompatibilityError(
            f"Model '{model.identifier}' hard-requires modalities "
            f"{model_missing} but benchmark '{benchmark.identifier}' does not "
            f"provide them. Benchmark provides: "
            f"{(bench_required | bench_accepted) or '{}'}."
        )

    # Check 3: Region mapping.
    required_region: Optional[str] = getattr(benchmark, 'region', None)
    if required_region and required_region not in model.region_layer_map:
        raise CompatibilityError(
            f"Model '{model.identifier}' has no layer mapping for region "
            f"'{required_region}'. Mapped regions: {set(model.region_layer_map.keys())}. "
            f"Use the Arena tool to explore and commit layer mappings."
        )

    # Check 4 (v1.5): Input/Output Catalog documentation conformance. Warn-only;
    # never changes pass/fail. Surfaces inputs or outputs that are not documented
    # in the catalog so undocumented channels are flagged at pre-flight.
    check_io_catalog(model, benchmark)


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
