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

- ``benchmark.required ⊆ model.available`` — model can process the
  benchmark's input format.
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

from .model_interface import UnifiedModel


class CompatibilityError(Exception):
    """Raised when a model cannot be scored on a benchmark."""
    pass


class CompatibilityWarning(UserWarning):
    """Issued when a model can run but will not use all available modalities."""
    pass


def check_compatibility(model: UnifiedModel, benchmark) -> None:
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

    # Check 2: Hard gate — model's required modalities must be a subset of
    # the benchmark's single input format (its required set). A locked-fusion
    # model that needs modalities the benchmark doesn't provide cannot run.
    model_missing = model_required - bench_required
    if model_missing:
        raise CompatibilityError(
            f"Model '{model.identifier}' hard-requires modalities "
            f"{model_missing} but benchmark '{benchmark.identifier}' does not "
            f"provide them. Benchmark provides: {bench_required or '{}'}."
        )

    # Check 3: Region mapping.
    required_region: Optional[str] = getattr(benchmark, 'region', None)
    if required_region and required_region not in model.region_layer_map:
        raise CompatibilityError(
            f"Model '{model.identifier}' has no layer mapping for region "
            f"'{required_region}'. Mapped regions: {set(model.region_layer_map.keys())}. "
            f"Use the Arena tool to explore and commit layer mappings."
        )
