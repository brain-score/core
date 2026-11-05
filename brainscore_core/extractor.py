"""The extractor protocol: what records activity for one input modality.

An extractor owns input preparation, the forward pass, hooks, caching and
assembly packaging. ``BrainScoreModel(extractor={modality: extractor})``
takes one per modality; the keys declare the modalities the subject takes in.
"Wrapper" is the name of the extractors Brain-Score ships (``TextWrapper``,
``VisionWrapper``, ...). Nothing needs to subclass this protocol.
"""

from typing import Any, Protocol, Sequence


class Extractor(Protocol):
    """Callable ``(stimuli, layers) -> NeuroidAssembly``.

    ``identifier`` and ``backbone_id`` key the activation cache (registrations
    that share weights share ``backbone_id``). An extractor that keeps state
    between calls also defines ``reset()``.

    An extractor may also define ``layer_names()``, returning every path it
    can record, relative to the module it wraps. ``BrainScoreModel`` then
    rejects an unknown ``region_layer_map`` path or raw recording target
    before any forward pass; a ``':input'`` site is checked by its module
    path. An extractor without ``layer_names()`` skips this early check.
    """

    identifier: str

    def __call__(self, stimuli: Any, layers: Sequence[str]) -> Any:
        ...
