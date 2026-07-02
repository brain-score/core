"""Input/output event dataclasses for the Brain-Score subject contract."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:  # annotation-only; resolved against the vendored brainio
    from brainscore_core.supported_data_standards.brainio.assemblies import (
        NeuroidAssembly, BehavioralAssembly)
    from .multimodal import MultimodalStimulusSet
    from .selection import UnitSelection


@dataclass
class Selection:
    """Identifies which units inside a model to perturb.

    The simplest case is a layer name + optional unit indices. Future
    extensions (functional localizers, contrast-based selection, mask arrays)
    add more fields without changing the dataclass shape.

    :param layer: Dotted module path within the model, e.g.
        ``'language_model.layers.20'`` or ``'visual.transformer.blocks.10'``.
        Resolved by the registered ``state_change_fn`` against the concrete
        model object.
    :param indices: Optional list of unit indices within the layer to
        perturb. ``None`` means ALL units at the layer.
    :param metadata: Free-form bag for selection criteria the
        ``state_change_fn`` can interpret (e.g., a localizer mask, a
        functional contrast specification).
    """
    layer: str
    indices: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Perturbation:
    """How to modify the activations at the selected units.

    :param kind: One of ``'zero'`` (zero out), ``'scale'`` (multiply by
        ``scale``), ``'replace'`` (replace with ``replacement`` tensor),
        or model-specific kinds (e.g., ``'noise'``) that the registered
        ``state_change_fn`` knows how to interpret.
    :param scale: Multiplier for ``kind='scale'``. Ignored otherwise.
    :param replacement: Tensor or array to substitute in for ``kind='replace'``.
        Shape must match the selected units' activations.
    """
    kind: str
    scale: float = 0.0
    replacement: Optional[Any] = None


@dataclass
class StateChange:
    """Induce a state change in the model (dysfunction, lesion, perturbation).

    Used to simulate conditions like dyslexia, prosopagnosia, or pharmacological
    effects. Passed to ``model.process()`` the same way stimuli are — the
    interface is generic over input event type so benchmarks studying neural
    dysfunction or drug effects do not need a separate model method.

    Two roles, distinguished by ``kind``:

    1. **Apply** a perturbation: ``kind`` names a perturbation class
       (``'ablation'``, ``'lesion'``, ``'pharmacological'``, ...). The
       ``state_change_fn`` registered on the model installs the perturbation
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
    target: Optional[Union['Selection', 'UnitSelection']] = None
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


@dataclass
class CameraFrame:
    """A single camera view at one timestep.

    Schema follows the DROID dataset convention (180x320 RGB stereo + wrist);
    extensible via optional depth and intrinsics for richer setups. Cameras are
    keyed by name on :class:`EnvironmentStep` (e.g., ``'exterior_1'``,
    ``'exterior_2'``, ``'wrist'``).

    v1.5 deprecation: this is a robotics-specific type. New code should import it
    from the robotics environment harness in the ``brainscore`` package and pack
    it into ``EnvironmentStep.observation``. It is kept in ``core`` for one
    release so existing embodied registrations keep working.
    """
    rgb: Any  # numpy.ndarray (H, W, 3) uint8 — required
    depth: Optional[Any] = None  # numpy.ndarray (H, W) float32, in meters
    intrinsics: Optional[Any] = None  # numpy.ndarray (3, 3) float64
    extrinsics: Optional[Any] = None  # numpy.ndarray (4, 4) float64, world ← camera


@dataclass
class Proprioception:
    """Robot proprioceptive state.

    Field names and shapes mirror the DROID schema (Franka 7-DOF arm + parallel
    gripper) so existing DROID-format checkpoints can be evaluated without a
    coordinate transform. The ``base_*`` fields are unused for arm-only setups;
    populate them for mobile-manipulation robots.

    v1.5 deprecation: this is a robotics-specific type. New code should import it
    from the robotics environment harness in the ``brainscore`` package and pack
    it into ``EnvironmentStep.observation``. It is kept in ``core`` for one
    release so existing embodied registrations keep working.
    """
    joint_position: Any  # (n_joints,) float64 — 7 for Franka
    cartesian_position: Any  # (6,) float64 — xyz + euler/rpy
    gripper_position: Any  # (1,) float64 — 0 (open) ↔ 1 (closed)
    joint_velocity: Optional[Any] = None  # (n_joints,) float64
    cartesian_velocity: Optional[Any] = None  # (6,) float64
    gripper_velocity: Optional[Any] = None  # (1,) float64
    # Mobile-manipulation extensions (None for arm-only platforms):
    base_position: Optional[Any] = None  # (3,) float64 — x, y, theta in odom frame
    base_velocity: Optional[Any] = None  # (3,) float64 — linear x, linear y, angular z


@dataclass
class EnvironmentStep:
    """Single step of an embodied environment.

    Used for interactive evaluation where the model observes, acts, and receives
    the environment's response over multiple ticks. Passed to ``process()`` like
    a stimulus — the interface generalizes over input type rather than adding
    a separate agent method.

    v1.5: this is a device-agnostic envelope. The preferred field is
    ``observation`` (a harness-defined payload). The ``cameras`` and
    ``proprioception`` fields are the deprecated DROID-shaped instantiation,
    kept for one release; the robotics environment harness in the ``brainscore``
    package is their going-forward home. ``core`` commits only to the
    observation-in / action-out envelope, not to any device's field layout.

    Schema follows DROID (https://droid-dataset.github.io/) for arm
    manipulation. Mobile-manipulation extensions are additive: populate
    ``proprioception.base_*`` and any base-mounted cameras under ``cameras``.

    Episode control signals (``is_first``, ``is_last``, ``is_terminal``,
    ``reward``, ``discount``) follow the RLDS convention so existing
    DROID-format demonstration data can be replayed through ``process()``
    one step at a time.

    Examples::

        # Single arm manipulation step (DROID-shaped):
        EnvironmentStep(
            cameras={'exterior_1': CameraFrame(rgb=img1),
                     'exterior_2': CameraFrame(rgb=img2),
                     'wrist': CameraFrame(rgb=wrist_img)},
            proprioception=Proprioception(
                joint_position=q, cartesian_position=ee_pose,
                gripper_position=g),
            instruction="put the red block in the bowl",
            step_num=0, is_first=True,
        )

        # Mobile manipulation step (adds base state + a base-mounted camera):
        EnvironmentStep(
            cameras={'wrist': ..., 'exterior_1': ..., 'base_front': ...},
            proprioception=Proprioception(..., base_position=[x, y, theta]),
            instruction="pick up the cup from the table by the door",
            step_num=42,
        )
    """
    # Deprecated DROID-shaped fields, now optional and kept for one release.
    # New code uses the generic ``observation`` field plus the robotics harness;
    # these remain so existing embodied registrations keep working unchanged.
    cameras: Optional[Dict[str, 'CameraFrame']] = None
    proprioception: Optional['Proprioception'] = None
    instruction: Optional[str] = None
    step_num: int = 0
    # RLDS / DROID episode control signals (optional for non-episodic eval):
    is_first: bool = False
    is_last: bool = False
    is_terminal: bool = False
    reward: Optional[float] = None
    discount: Optional[float] = None
    # Free-form context — benchmarks that need to thread custom state (previous
    # actions, object IDs, scene graph, etc.) without growing the schema.
    context: Dict[str, Any] = field(default_factory=dict)
    # The perceptual content of this tick: what the agent observes. Canonically
    # a (Multimodal)StimulusSet — an EnvironmentStep PRESENTS a stimulus, it does
    # not sit *beside* StimulusSet in the input taxonomy; it wraps one plus the
    # closed-loop/episode context. The Dict[str, Any] member is the harness-
    # defined envelope (robotics, Atari, browser, grid game) used today, kept
    # permissive so core commits only to "an observation in, an action out".
    observation: Optional[Union[
        'MultimodalStimulusSet', Dict[str, Any], 'Message',
    ]] = None

    def __post_init__(self):
        # Bridge the deprecated DROID fields onto the generic observation so the
        # dispatch path is uniform: if no observation was given but the DROID
        # fields were, expose them as the observation.
        if self.observation is None and (
            self.cameras is not None or self.proprioception is not None
        ):
            self.observation = {
                "cameras": self.cameras,
                "proprioception": self.proprioception,
            }


@dataclass
class EnvironmentResponse:
    """The model's response to one :class:`EnvironmentStep`.

    v1.5: ``action`` is a harness-defined payload, not pinned to any device. The
    environment harness decides its shape. The DROID instantiation uses a compact
    7-D action (6 joint velocities + 1 gripper position); other environments use
    their own. ``action_dict`` carries an optional structured form and
    ``metadata`` carries telemetry (value estimate, attention maps, predicted
    reward) without growing the schema.
    """
    action: Any  # numpy.ndarray (action_dim,) float64
    action_dict: Optional[Dict[str, Any]] = None  # DROID-style structured action
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """A communicative event that is valid as BOTH an output and an input.

    ``Message`` is the symmetry-closing member of the I/O unions: the first type
    a model can *emit* (``OutputEvent``) that another model can directly
    *consume* (``InputEvent``). That dual membership is exactly what
    first-class multi-agent social interaction needs — one agent's utterance is
    the next agent's stimulus, with no bespoke routing protocol and no ABC
    change (the design's named multi-agent enabler; see Developer Reference
    §6.1 item 3).

    ``process(Message)`` routes to the model's ``action_fn`` (an agent
    *responds* to a message), and a responder may return either a ``Message`` or
    an :class:`EnvironmentResponse`. ``content`` is the payload (text, an action,
    a structured proposal); ``sender`` / ``recipient`` carry addressing for
    multi-party settings; ``metadata`` carries telemetry without growing the
    schema.
    """
    content: Any
    sender: Optional[str] = None
    recipient: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def output_event_kind(output: 'OutputEvent') -> str:
    """Classify an ``OutputEvent`` member so metrics can dispatch on output type
    instead of the benchmark hard-selecting one (Developer Reference §6.1 item 4).

    Returns one of ``'neural'`` (NeuroidAssembly), ``'behavioral'``
    (BehavioralAssembly), ``'environment'`` (EnvironmentResponse), ``'message'``
    (Message), ``'perturbation'`` (PerturbationApplied), or ``'unknown'``. Uses
    duck typing for the two forward-ref assembly types (whose concrete classes
    aren't imported in core) so it never requires a hard BrainIO import.
    """
    if isinstance(output, EnvironmentResponse):
        return 'environment'
    if isinstance(output, Message):
        return 'message'
    if isinstance(output, PerturbationApplied):
        return 'perturbation'
    cls_names = {c.__name__ for c in type(output).__mro__}
    if 'NeuroidAssembly' in cls_names:
        return 'neural'
    if 'BehavioralAssembly' in cls_names:
        return 'behavioral'
    return 'unknown'


def dispatch_metric(output: 'OutputEvent', metric_map: Dict[str, Any]):
    """Pick a metric for an output by its :func:`output_event_kind`.

    ``metric_map`` maps a kind (``'neural'`` / ``'behavioral'`` / ``'message'`` /
    …) to a metric callable. Raises ``KeyError`` with the available kinds when an
    output's kind isn't registered — so a benchmark declares the metrics it
    supports by output type and the dispatch is data-driven, not hard-coded.
    """
    kind = output_event_kind(output)
    if kind not in metric_map:
        raise KeyError(
            f"no metric registered for output kind {kind!r}; "
            f"metric_map has {sorted(metric_map)}")
    return metric_map[kind]


# Input event type for process().
#
# The union spans two orthogonal axes, flattened here for single-dispatch:
#   - PAYLOAD axis (what is presented): StimulusSet / MultimodalStimulusSet.
#   - PROTOCOL axis (mode of engagement): perceive (open-loop batch readout),
#     interact (closed-loop, EnvironmentStep), perturb (intervention, StateChange).
# Read this union as "modes of engagement": a bare StimulusSet is the degenerate
# *perceive* case (passed directly for ergonomics — the 99% path); StateChange is
# a pure intervention that carries no stimulus; EnvironmentStep *contains* a
# stimulus (its `observation` is a MultimodalStimulusSet plus episode context),
# so it WRAPS a payload rather than being a sibling of one. A fully symmetric
# spec would wrap the perceive case too (Perceive(stimulus)) and pair each input
# mode with an OutputEvent member; that refactor is deferred (see the design docs)
# until a second interactive modality makes the flattening costly.
# StimulusSet is a forward ref to avoid a hard BrainIO dependency in core; the
# runtime type check happens at dispatch time inside BrainScoreModel.process().
InputEvent = Union['StimulusSet', StateChange, EnvironmentStep, Message]  # type: ignore[name-defined]

# Output event type for process(). Symmetric with InputEvent: every process()
# call returns one of these members, and the union grows by ADDING a member when
# a benchmark needs a new output shape — never by changing the Subject ABC. This
# is the output-side answer to the same extensibility argument that motivated
# InputEvent (Addressing Stakeholder Comments §2.3, Martin Schrimpf).
#
# Three of the four members exist concretely today:
#   - NeuroidAssembly    — neural recording (sensory-stimulus path); forward ref
#                          so core stays free of a hard BrainIO import
#   - BehavioralAssembly — behavioral readout / generation; forward ref likewise
#   - EnvironmentResponse — one embodied step (the action/trajectory member)
#   - PerturbationApplied — state-change acknowledgement + reset handle
# Future members slot in here when their first benchmark lands, with no ABC
# change: MotorOutput (continuous motor regression, §2.4) and GeneratedSequence
# (variable-length tokens + per-token logprobs).
OutputEvent = Union['NeuroidAssembly', 'BehavioralAssembly',
                    EnvironmentResponse, PerturbationApplied, Message]
