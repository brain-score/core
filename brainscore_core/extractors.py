"""Activity extraction is a callable contract, independent of a model framework."""

from typing import Protocol, Sequence

from .supported_data_standards.brainio.assemblies import NeuroidAssembly
from .supported_data_standards.brainio.stimuli import StimulusSet


class Extractor(Protocol):
    """Own input preparation, execution, recording, caching and assembly packaging."""

    identifier: str
    backbone_id: str

    def __call__(self, stimuli: StimulusSet, *, layers: Sequence[str]) -> NeuroidAssembly:
        """Record the requested layer outputs in stimulus order."""
        ...


class ResettableExtractor(Extractor, Protocol):
    """Optional lifecycle for extractors that keep state between calls."""

    def reset(self) -> None:
        """Clear episode state while keeping model weights and configuration."""
        ...
