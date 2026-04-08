"""
Pre-flight compatibility checker for Brain-Score.

Validates model-benchmark compatibility before any computation begins.
Two-tier modality validation: required (hard gate) vs available (soft warning).
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
    """
    Fast pre-flight check. Raises CompatibilityError if the model
    clearly cannot run on this benchmark. Issues CompatibilityWarning
    if the model can run but will degrade (missing optional modalities).

    Two-tier check:
    1. required_modalities (hard gate): model must support ALL of these.
    2. available_modalities (soft): model CAN use these but doesn't have to.
    3. region check: model must have a mapping for the benchmark's target region.
    """
    # Check 1: Hard gate on required modalities
    required: Set[str] = getattr(benchmark, 'required_modalities', set())
    missing_required = required - model.supported_modalities
    if missing_required:
        raise CompatibilityError(
            f"Model '{model.identifier}' does not support modalities required by "
            f"benchmark '{benchmark.identifier}': {missing_required}. "
            f"Model supports: {model.supported_modalities}."
        )

    # Check 2: Soft check on available modalities
    available: Set[str] = getattr(benchmark, 'available_modalities', set())
    unused = available - model.supported_modalities
    if unused:
        warnings.warn(
            f"Model '{model.identifier}' does not support optional modalities "
            f"provided by benchmark '{benchmark.identifier}': {unused}. "
            f"The model will run using only: "
            f"{model.supported_modalities & (required | available)}. "
            f"Scores may differ from a model that uses all modalities.",
            CompatibilityWarning,
        )

    # Check 3: Region mapping
    required_region: Optional[str] = getattr(benchmark, 'region', None)
    if required_region and required_region not in model.region_layer_map:
        raise CompatibilityError(
            f"Model '{model.identifier}' has no layer mapping for region "
            f"'{required_region}'. Mapped regions: {set(model.region_layer_map.keys())}. "
            f"Use the Arena tool to explore and commit layer mappings."
        )
