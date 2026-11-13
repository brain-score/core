"""Input/output event dataclasses for the Brain-Score subject contract."""

from .streaming import StreamEvent
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Union

if TYPE_CHECKING:  # annotation-only; resolved against the vendored brainio
    from brainscore_core.supported_data_standards.brainio.assemblies import (
        NeuroidAssembly, BehavioralAssembly)


@dataclass
class Selection:
    """Identifies which units inside a model to perturb.

    The simplest case is a layer name + optional unit indices. Future
    extensions (functional localizers, contrast-based selection, mask arrays)
    add more fields without changing the dataclass shape.

    :param layer: Dotted module path within the model, e.g.
        ``'language_model.layers.20'`` or ``'visual.transformer.blocks.10'``.
        Resolved by the model's registered ``intervention`` against the
        concrete model object. A path names the module's output; add ``':input'`` to
        name its first input instead, e.g.
        ``'model.layers.14.mlp.down_proj:input'`` for a transformer's MLP
        neurons, which are computed inline and output by no module. Recording
        uses the same names, so a region and a perturbation that share a
        string address the same tensor.
    :param indices: Optional list of unit indices within the layer to
        perturb. ``None`` means ALL units at the layer.
    :param axis: Tensor axis addressed by indices; -1 keeps the last-axis
        convention. None flattens all axes after the batch axis in row-major
        order, matching unpooled activation extraction.
    :param expected_shape: Optional non-batch tensor shape, checked before use.
        Localizers set this when coordinates describe a flattened output.
    :param metadata: Free-form bag for selection criteria the
        ``intervention`` can interpret (e.g., a localizer mask, a
        functional contrast specification).
    """
    layer: str
    indices: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    axis: Optional[int] = -1
    expected_shape: Optional[tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if self.axis is not None and (isinstance(self.axis, bool) or not isinstance(self.axis, int)):
            raise TypeError('Selection.axis must be an integer or None (flatten non-batch axes).')
        if self.axis == 0:
            raise ValueError('Selection cannot index the batch axis.')
        if self.expected_shape is not None:
            self.expected_shape = tuple(self.expected_shape)



@dataclass
class Perturbation:
    """How to modify the activations at the selected units.

    :param kind: One of ``'zero'`` (zero out), ``'scale'`` (multiply by
        ``scale``), ``'drive'`` (add ``amount`` — a stimulation / excitation
        analog), ``'replace'`` (replace with ``replacement`` tensor), or
        model-specific kinds (e.g., ``'noise'``) that the registered
        ``intervention`` knows how to interpret.
    :param scale: Multiplier for ``kind='scale'``. Ignored otherwise.
    :param amount: Additive offset for ``kind='drive'`` — inject activity into
        the selected units, the counterpart of ablation. A number, or (with
        ``indices=None``) an array or tensor broadcastable to the layer's
        activations, e.g. a direction to push along. Ignored otherwise.
    :param replacement: Tensor or array to substitute in for ``kind='replace'``.
        Shape must match the selected units' activations.
    """
    kind: str
    scale: float = 0.0
    replacement: Optional[Any] = None
    amount: Union[float, Any] = 0.0  # appended last to preserve the (kind, scale, replacement) positional API


@dataclass
class StateChange:
    """Apply or remove an intervention in the model.

    Used to simulate conditions like dyslexia, prosopagnosia, or pharmacological
    effects. A session (or the typed ``apply_state_change`` helper) delivers it
    to the subject; ``BrainScoreModel.process()`` also accepts it as a typed
    convenience, the same way it accepts stimuli. Benchmarks studying neural
    dysfunction or drug effects do not need a separate model method.

    Two roles, distinguished by ``kind``:

    1. **Apply** a perturbation: ``kind`` names a perturbation class
       (``'ablation'``, ``'lesion'``, ``'pharmacological'``, ...). The
       ``intervention`` registered on the model installs the perturbation
       (e.g., a forward hook), and ``process(StateChange)`` returns a
       :class:`PerturbationApplied` confirmation containing a ``handle_id``.
    2. **Remove** a previously-applied perturbation: ``kind='reset'`` and
       ``handle_id`` set to a previous PerturbationApplied's id. Removes
       just that perturbation; ``model.reset()`` removes ALL active ones.

    Examples::

        # Apply: zero out 100 units at a specific layer
        sc = StateChange(
            kind='ablation',
            target=Selection(layer='language_model.layers.20',
                             indices=list(range(100))),
            perturbation=Perturbation(kind='zero'),
        )
        applied = model.process(sc)
        # ... later:
        model.reset()  # clears all active perturbations
        # ... OR remove just this one:
        model.process(StateChange(kind='reset', handle_id=applied.handle_id))
    """
    kind: str
    target: Optional[Union['Selection', Any]] = None
    perturbation: Optional['Perturbation'] = None
    handle_id: Optional[str] = None  # for kind='reset', identifies what to undo
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerturbationApplied:
    """Returned from ``process(StateChange)`` after a perturbation is installed.

    :param handle_id: Opaque identifier; pass to a future
        ``StateChange(kind='reset', handle_id=...)`` to undo just this
        perturbation, or call ``model.reset()`` to clear all active ones.
    :param target: The Selection that was perturbed.
    :param perturbation: The Perturbation that was applied.
    :param applied_at: Step counter or wall-clock timestamp when applied.
        Useful for benchmarks that want to log perturbation timelines.
    """
    handle_id: str
    target: 'Selection'
    perturbation: 'Perturbation'
    applied_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# Input event type for process().
#
# The union spans two orthogonal axes, flattened here for single-dispatch:
#   - PAYLOAD axis (what is presented): a StimulusSet (a table, one row per moment).
#   - PROTOCOL axis (mode of engagement): perceive (open-loop batch readout),
#     interact (closed-loop: a row, a mapping of channel -> payload), perturb
#     (intervention, StateChange).
# A bare StimulusSet is the degenerate *perceive* case; StateChange is a pure
# intervention that carries no stimulus; a row is one closed-loop step. A message
# between models is a ``text`` event or row entry with the sender in ``meta``.
# StimulusSet is a forward ref to avoid a hard BrainIO dependency in core; the
# runtime type check happens at dispatch time inside BrainScoreModel.process().
InputEvent = Union['StimulusSet', StateChange, Mapping[str, Any], StreamEvent]  # type: ignore[name-defined]

# Output event type for process(). Symmetric with InputEvent: every process()
# call returns one of these members, and the union grows by ADDING a member when
# a benchmark needs a new output shape, never by changing the Subject ABC.
#   - NeuroidAssembly    - neural recording; forward ref so core stays free of a
#                          hard BrainIO import
#   - BehavioralAssembly - behavioral readout / generation; forward ref likewise
#   - PerturbationApplied - state-change acknowledgement + reset handle
#   - Mapping[str, Any]  - one closed-loop step's outputs: output channel -> payload
#   - StreamEvent        - one event on a channel
OutputEvent = Union['NeuroidAssembly', 'BehavioralAssembly',
                    PerturbationApplied, Mapping[str, Any], StreamEvent]
