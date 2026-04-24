"""
Pre-flight compatibility checker for Brain-Score.

Validates model-benchmark compatibility before any computation begins.
Two-tier modality validation on BOTH sides:

- Benchmark declares ``required_modalities`` (stimuli provide these and the
  model must be able to consume all of them) and ``available_modalities``
  (also present in stimuli but the model can skip them).
- Model declares ``required_modalities`` (benchmark must provide all of these
  — e.g. a locked-fusion model that cannot degrade) and
  ``available_modalities`` (all modalities the model can consume).

A pairing is valid when:
- ``benchmark.required ⊆ model.available`` (model can process mandatory stream)
- ``model.required ⊆ benchmark.required ∪ benchmark.available`` (benchmark
  actually provides everything the model hard-requires)
- ``benchmark.region ∈ model.region_layer_map`` (if the benchmark targets a
  neural region)

Missing optional modalities on either side surface as
``CompatibilityWarning``, not errors.
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

    Raises :class:`CompatibilityError` for hard mismatches; issues
    :class:`CompatibilityWarning` when the pairing will run but degrade.
    """
    model_available: Set[str] = set(model.available_modalities)
    model_required: Set[str] = set(model.required_modalities)

    # Check 1: Hard gate — benchmark's required modalities must ALL be
    # consumable by the model.
    bench_required: Set[str] = set(getattr(benchmark, 'required_modalities', set()))
    missing_required = bench_required - model_available
    if missing_required:
        raise CompatibilityError(
            f"Model '{model.identifier}' does not support modalities required by "
            f"benchmark '{benchmark.identifier}': {missing_required}. "
            f"Model available modalities: {model_available}."
        )

    # Check 2: Hard gate — model's required modalities must be in the
    # benchmark's provided streams (required ∪ available).
    bench_available: Set[str] = set(getattr(benchmark, 'available_modalities', set()))
    bench_provided = bench_required | bench_available
    model_missing = model_required - bench_provided
    if model_missing:
        raise CompatibilityError(
            f"Model '{model.identifier}' hard-requires modalities "
            f"{model_missing} but benchmark '{benchmark.identifier}' does not "
            f"provide them. Benchmark provides: {bench_provided or '{}'}."
        )

    # Check 3: Soft — benchmark-available modalities the model cannot use.
    unused = bench_available - model_available
    if unused:
        warnings.warn(
            f"Model '{model.identifier}' does not support optional modalities "
            f"provided by benchmark '{benchmark.identifier}': {unused}. "
            f"The model will run using only: "
            f"{model_available & bench_provided}. "
            f"Scores may differ from a model that uses all modalities.",
            CompatibilityWarning,
        )

    # Check 4: Region mapping.
    required_region: Optional[str] = getattr(benchmark, 'region', None)
    if required_region and required_region not in model.region_layer_map:
        raise CompatibilityError(
            f"Model '{model.identifier}' has no layer mapping for region "
            f"'{required_region}'. Mapped regions: {set(model.region_layer_map.keys())}. "
            f"Use the Arena tool to explore and commit layer mappings."
        )
