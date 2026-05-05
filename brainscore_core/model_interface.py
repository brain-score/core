"""
Unified model interface for Brain-Score.

Defines the core abstractions for model evaluation:
- TaskContext: benchmark-to-model communication
- StateChange: future-proof input event type for dysfunction/lesion/perturbation
- EnvironmentStep / EnvironmentResponse / CameraFrame / Proprioception:
  embodied-agent input event types (DROID-shaped schema)
- UnifiedModel: the single evaluation interface (ABC)
- BrainScoreModel: compositional implementation with preprocessors + shared
  activations_model + optional generation_fn / action_fn dispatch slots

## Input generalization (Martin Schrimpf feedback, April 2026)

`process()` takes an *input event*, not just stimuli. Implemented input types:

- **`StimulusSet`** — perceptual input (vision/text/audio/video). Primary path.
- **`EnvironmentStep`** — one tick of an embodied environment. DROID-shaped
  schema (multi-camera RGB + proprioception + instruction). Dispatches to the
  model's `action_fn` when registered. Spec follows
  https://droid-dataset.github.io/ for arm manipulation; mobile-manipulation
  extensions live on `Proprioception.base_*`.

Reserved (raises `NotImplementedError`):

- **`StateChange`** — induce a dysfunction/lesion/perturbation
  (e.g., dyslexia, prosopagnosia, pharmacological effects).

See [[Unified Model Interface - Vision and Goals]] §Input Generalization
and [[Unified Model Interface - Embodied I/O Reference]] for the DROID
schema citations and the mobile-manipulation roadmap.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


@dataclass
class TaskContext:
    """
    Everything a model needs to understand what task the benchmark wants.

    Benchmarks populate all relevant fields. Models consume whichever
    fields match their paradigm:
    - Feature models use fitting_stimuli + label_set to train a readout
    - Instruction-following models use instruction + label_set to build a prompt
    - Models that support both can choose their preferred approach

    ``prefer_path`` lets a benchmark (or an experimenting user) pin which
    behavioral dispatch path is selected when a model supports more than
    one. Accepts ``'auto'`` (default — generation if available, else
    readout), ``'generation'`` (force generation, error if generation_fn
    is missing or instruction is empty), or ``'readout'`` (force readout,
    error if behavioral_readout_layer is missing or fitting_stimuli is
    empty). This is the public API replacement for the legacy monkey-patch
    pattern of setting ``model._generation_fn = None`` to force readout.
    """
    task_type: str
    label_set: Optional[List[str]] = None
    fitting_stimuli: Optional[Any] = None
    instruction: Optional[str] = None
    prefer_path: Optional[str] = None  # 'auto' | 'generation' | 'readout'
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    target: Optional['Selection'] = None
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

    Schema follows the DROID dataset convention (180×320 RGB stereo + wrist);
    extensible via optional depth and intrinsics for richer setups. Cameras are
    keyed by name on :class:`EnvironmentStep` (e.g., ``'exterior_1'``,
    ``'exterior_2'``, ``'wrist'``).
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
    cameras: Dict[str, 'CameraFrame']
    proprioception: 'Proprioception'
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


@dataclass
class EnvironmentResponse:
    """The model's response to one :class:`EnvironmentStep`.

    Mirrors DROID's compact ``action`` (default 7-D: 6 joint velocities + 1
    gripper position) plus an optional structured ``action_dict`` for richer
    control modes (Cartesian deltas, base velocity, etc.). ``metadata`` lets
    models report telemetry (value estimate, attention maps, predicted reward)
    without growing the action schema.
    """
    action: Any  # numpy.ndarray (action_dim,) float64
    action_dict: Optional[Dict[str, Any]] = None  # DROID-style structured action
    metadata: Dict[str, Any] = field(default_factory=dict)


# Input event type for process().
# The interface is generic over input event type. StimulusSet is imported
# lazily to avoid a hard dependency on the BrainIO types from core. In type
# hints this is represented by Any; the actual type check happens at
# dispatch time inside BrainScoreModel.process().
InputEvent = Union['StimulusSet', StateChange, EnvironmentStep]  # type: ignore[name-defined]


class UnifiedModel(ABC):
    """
    Base interface for all Brain-Score models.

    Defines identity, lifecycle, and a single processing method.
    Benchmarks call process(stimuli) for all evaluation. What the model
    perceives is determined by the stimulus content. What the model
    produces is determined by the measurement configuration (start_task
    for behavioral, start_recording for neural).

    ## Two-tier modality declaration

    Models declare modality support at two tiers, mirroring the benchmark-side
    contract (`required_modalities` / `available_modalities`):

    - ``required_modalities``: HARD requirement. The stimulus set MUST contain
      columns covering every modality in this set, or the compatibility check
      rejects the pairing before any compute runs. Use this for models that
      cannot produce a prediction at all without a particular input modality
      (pure language models, pure video models, and models whose forward pass
      requires all of vision+audio+text fused together, e.g. TRIBEv2-locked).
    - ``available_modalities``: SOFT capability. Modalities the model CAN
      consume if provided, but does not hard-require. A benchmark that
      provides additional available modalities will surface a
      ``CompatibilityWarning`` for modalities present in the stimuli that the
      model cannot consume.

    The invariant ``required_modalities ⊆ available_modalities`` always holds.
    The legacy ``supported_modalities`` property is kept as an alias for
    ``available_modalities`` so older benchmarks continue to work.
    """

    @property
    @abstractmethod
    def identifier(self) -> str:
        ...

    @property
    @abstractmethod
    def region_layer_map(self) -> Dict[str, str]:
        ...

    @property
    @abstractmethod
    def supported_modalities(self) -> Set[str]:
        """All modalities the model can consume. Concrete subclasses MUST
        implement this. :py:attr:`available_modalities` defaults to this
        value so code written for the two-tier contract works unchanged on
        pre-two-tier subclasses."""
        ...

    @property
    def available_modalities(self) -> Set[str]:
        """All modalities the model can consume. Defaults to
        :py:attr:`supported_modalities`. Concrete subclasses may override
        this directly (preferred new API) or rely on the default."""
        return self.supported_modalities

    @property
    def required_modalities(self) -> Set[str]:
        """Modalities the model HARD-requires in stimuli. Empty by default —
        most models can run on whichever modality the benchmark provides, as
        long as the intersection with :py:attr:`available_modalities` is
        non-empty. Override for models that must receive a specific modality
        (single-modality text/video backbones, multimodal-fusion models that
        cannot degrade, etc.)."""
        return set()

    @abstractmethod
    def process(self, stimuli) -> Any:
        ...

    def start_task(self, task_context: TaskContext) -> None:
        self._task_context: Optional[TaskContext] = task_context

    def start_recording(self,
                        recording_target: Union[str, List[str]],
                        time_bins: Optional[List[Tuple[int, int]]] = None,
                        recording_type: Optional[str] = None) -> None:
        pass

    def reset(self) -> None:
        pass


class BrainScoreModel(UnifiedModel):
    """
    Compositional implementation of UnifiedModel.

    Composes preprocessors (simple callables, one per modality) with a
    shared activations_model (e.g. PytorchWrapper) that handles forward
    pass, hook-based layer extraction, caching, and NeuroidAssembly
    packaging. Preprocessing is the only modality-specific part; the
    activations_model is shared across all modalities.

    process() follows a single path:
    1. Detect modality from stimulus columns
    2. Route to the activations_model for vision extraction, or to the
       preprocessor callable for other modalities (temporary until
       TextWrapper provides symmetric extraction)
    3. Return NeuroidAssembly
    """

    COLUMN_TO_MODALITY: Dict[str, str] = {
        'image_file_name': 'vision',
        'image_path': 'vision',
        'filename': 'vision',
        'sentence': 'text',
        'text': 'text',
        'video_path': 'video',
        'audio_path': 'audio',
        'audio_file_name': 'audio',
        'audio_file': 'audio',
    }

    def __init__(
        self,
        identifier: str,
        model: Any,
        region_layer_map: Dict[str, str],
        preprocessors: Dict[str, Callable],
        activations_model: Any = None,
        visual_degrees: int = 8,
        behavioral_readout_layer: Optional[str] = None,
        generation_fn: Optional[Callable] = None,
        action_fn: Optional[Callable] = None,
        state_change_fn: Optional[Callable] = None,
        required_modalities: Optional[Set[str]] = None,
        backbone_id: Optional[str] = None,
    ) -> None:
        """
        :param generation_fn: Optional callable for instruction-following
            behavioral tasks. Signature:
                generation_fn(stimulus_row, instruction, label_set) -> str
            Must return one of the strings in ``label_set``. When provided,
            a TaskContext that carries both `instruction` and `label_set`
            will route behavioral evaluation through this function instead
            of the logistic readout. This lets instruction-following VLMs
            (Qwen-VL, BLIP-2 decoder, etc.) answer via generation+parsing
            while feature models (CLIP, ResNet) stay on the readout path.
        :param action_fn: Optional callable for embodied evaluation.
            Signature:
                action_fn(env_step: EnvironmentStep) -> EnvironmentResponse
            When provided, ``process(EnvironmentStep)`` dispatches to this
            function. The model is responsible for selecting an action given
            the observation; the benchmark is responsible for stepping the
            environment with the returned action and constructing the next
            ``EnvironmentStep``. Use this for DROID-shaped manipulation
            policies, mobile-manipulation policies, or any closed-loop
            agent. See the ``EnvironmentStep`` / ``EnvironmentResponse``
            dataclasses for the I/O schema.
        :param state_change_fn: Optional callable for perturbation evaluation
            (lesions, ablations, pharmacological effects). Signature:
                state_change_fn(state_change: StateChange) ->
                    Tuple[PerturbationApplied, Callable[[], None]]
            The callable installs the perturbation (typically via a
            forward-hook on the underlying model) and returns the
            ``PerturbationApplied`` confirmation paired with a cleanup
            closure. ``BrainScoreModel`` retains the cleanup keyed by
            ``handle_id`` and invokes it on ``model.reset()`` or on
            ``process(StateChange(kind='reset', handle_id=...))``. When
            ``state_change_fn`` is ``None``, ``process(StateChange)``
            raises ``NotImplementedError``.
        :param required_modalities: Optional set of modalities this model
            HARD-requires the benchmark to provide. Must be a subset of
            ``preprocessors.keys()``. Defaults to empty (soft capability for
            all modalities). Set for single-modality backbones (text, video)
            and multimodal-fusion models that cannot degrade to a subset.
        :param backbone_id: Optional cache-key identifier, propagated to
            preprocessors/activations_model that expose a ``backbone_id``
            attribute. When two registrations share underlying backbone
            weights (e.g. BLIP-2 and a future InstructBLIP both using the
            same ViT-G), passing the same ``backbone_id`` lets the
            ``@store_xarray`` cache entries be reused across registrations.
            Defaults to ``identifier`` for backwards compatibility.
        """
        self._identifier_str = identifier
        self._model = model
        self._region_layer_map_dict = region_layer_map
        self._preprocessors = preprocessors
        self._activations_model = activations_model
        self._visual_degrees_val = visual_degrees
        self._behavioral_readout_layer = behavioral_readout_layer
        self._generation_fn = generation_fn
        self._action_fn = action_fn
        self._state_change_fn = state_change_fn
        # handle_id -> cleanup callable. Populated when process(StateChange)
        # successfully installs a perturbation; drained by model.reset() or
        # by process(StateChange(kind='reset', handle_id=...)).
        self._active_perturbations: Dict[str, Callable[[], None]] = {}
        self._recording_layer: Optional[str] = None
        # Multi-region recording state. Populated by start_recording when
        # called with a list of regions; left empty for single-region calls.
        # _is_multi_region gates whether process() tags neuroids with a
        # 'region' coord — keeping single-region output bit-for-bit identical
        # to pre-multi-region behavior.
        self._recording_regions: List[str] = []
        self._recording_layers: List[str] = []
        self._is_multi_region: bool = False
        self._task_context: Optional[TaskContext] = None
        # Lazily instantiated the first time a behavioral task is started.
        self._readout_classifier = None  # type: ignore[assignment]
        # When True, process() returns generated predictions from generation_fn
        # rather than feature-based probabilities or activations.
        self._use_generation_for_task: bool = False

        required = set(required_modalities) if required_modalities else set()
        available = set(preprocessors.keys())
        if not required.issubset(available):
            raise ValueError(
                f"required_modalities {required} must be a subset of the "
                f"available preprocessors {available}. A model cannot require "
                f"a modality for which it has no preprocessor."
            )
        self._required_modalities = required
        self._backbone_id = backbone_id or identifier

    @property
    def identifier(self) -> str:
        return self._identifier_str

    @property
    def region_layer_map(self) -> Dict[str, str]:
        return dict(self._region_layer_map_dict)

    @property
    def supported_modalities(self) -> Set[str]:
        return set(self._preprocessors.keys())

    @property
    def available_modalities(self) -> Set[str]:
        # Explicit override (identical to supported_modalities) so the
        # two-tier API is anchored on BrainScoreModel rather than inherited
        # from the legacy alias.
        return set(self._preprocessors.keys())

    @property
    def required_modalities(self) -> Set[str]:
        return set(self._required_modalities)

    @property
    def backbone_id(self) -> str:
        """Cache-key identifier. Two registrations that share backbone
        weights can opt into shared @store_xarray cache by constructing
        with the same ``backbone_id``. Defaults to ``identifier``."""
        return self._backbone_id

    # When a stimulus set carries columns for multiple modalities (e.g. a
    # word image *and* the word's text string), pick one deterministically.
    # Vision wins over text — it's the richer signal and matches the
    # paradigm of benchmarks that were designed image-first.
    MODALITY_PRIORITY: Tuple[str, ...] = ('vision', 'text', 'audio', 'video')

    def _detect_modalities(self, stimuli) -> Set[str]:
        detected: Set[str] = set()
        for col in stimuli.columns:
            modality = self.COLUMN_TO_MODALITY.get(col)
            if modality and modality in self._preprocessors:
                detected.add(modality)
        return detected

    def _pick_modality(self, detected: Set[str]) -> str:
        """Return a deterministic modality from the detected set, using
        MODALITY_PRIORITY as a tiebreaker. Models that only support one
        modality end up with a single-element set and this is a no-op."""
        for m in self.MODALITY_PRIORITY:
            if m in detected:
                return m
        return next(iter(detected))

    def process(self, input_event) -> Any:
        # Dispatch on input event type. StimulusSet (perceptual input) is the
        # primary path; EnvironmentStep dispatches to the registered action_fn
        # for embodied evaluation; StateChange dispatches to the registered
        # state_change_fn for perturbation/lesion/ablation evaluation.
        if isinstance(input_event, StateChange):
            return self._dispatch_state_change(input_event)
        if isinstance(input_event, EnvironmentStep):
            if self._action_fn is None:
                raise NotImplementedError(
                    f"Model '{self.identifier}' has no action_fn registered. "
                    f"Embodied evaluation requires the model to declare an "
                    f"action_fn(env_step) -> EnvironmentResponse callable at "
                    f"BrainScoreModel construction time. Received: "
                    f"EnvironmentStep(step_num={input_event.step_num})."
                )
            response = self._action_fn(input_event)
            if not isinstance(response, EnvironmentResponse):
                raise TypeError(
                    f"Model '{self.identifier}' action_fn returned "
                    f"{type(response).__name__}; expected EnvironmentResponse. "
                    f"Wrap the action in EnvironmentResponse(action=...) so "
                    f"benchmarks see a consistent shape across models."
                )
            return response

        # Perceptual input (StimulusSet).
        stimuli = input_event

        # Behavioral mode (generation path): model generates a label for
        # each stimulus via instruction + generation_fn.
        if (self._use_generation_for_task
                and self._task_context is not None
                and self._requires_behavioral_readout(self._task_context.task_type)):
            return self._generate_predictions(stimuli)

        # Behavioral mode (readout path): logistic classifier returns
        # probabilities over the label_set.
        if (self._readout_classifier is not None
                and self._task_context is not None
                and self._requires_behavioral_readout(self._task_context.task_type)):
            return self._predict_probabilities(stimuli)

        detected = self._detect_modalities(stimuli)

        if not detected:
            raise ValueError(
                f"No recognized modality columns in stimulus set. "
                f"Columns present: {list(stimuli.columns)}. "
                f"Known column mappings: {self.COLUMN_TO_MODALITY}. "
                f"Model supports: {self.supported_modalities}."
            )

        modality = self._pick_modality(detected)

        # Resolve which layers to extract. Prefer the multi-region list if
        # populated (set by start_recording with a list); otherwise fall
        # back to the singular _recording_layer for the backward-compatible
        # single-region path.
        if self._recording_layers:
            layers = list(self._recording_layers)
        elif self._recording_layer:
            layers = [self._recording_layer]
        else:
            layers = []

        # Vision path: delegate to activations_model (PytorchWrapper handles
        # preprocessing, forward pass, hooks, caching, and assembly packaging)
        if modality == 'vision' and self._activations_model is not None:
            assembly = self._activations_model(stimuli, layers=layers)
            if self._is_multi_region:
                assembly = self._tag_neuroids_with_regions(assembly)
            return assembly

        # Other modalities: check if the preprocessor is a full extractor
        # (like TextWrapper — accepts (stimuli, layers=[])) or a legacy
        # callable (accepts (model, stimuli, recording_layer=...)).
        # Duck-type: extractors have an `identifier` attribute.
        preprocessor = self._preprocessors[modality]
        if hasattr(preprocessor, 'identifier'):
            assembly = preprocessor(stimuli, layers=layers)
            if self._is_multi_region:
                assembly = self._tag_neuroids_with_regions(assembly)
            return assembly
        return preprocessor(
            self._model, stimuli,
            recording_layer=self._recording_layer,
        )

    def _tag_neuroids_with_regions(self, assembly):
        """Add a 'region' coord to the neuroid axis of a multi-layer assembly.

        For each layer in the assembly's neuroid coord, look up which
        region(s) the active recording_regions mapped that layer to.
        Multiple regions sharing a layer get joined with '|' (xarray
        coords don't natively support per-element lists in the 2022.3
        pin we hold).
        """
        import numpy as np
        if 'layer' not in assembly.coords:
            return assembly
        layer_to_regions: Dict[str, List[str]] = {}
        for region in self._recording_regions:
            layer = self._region_layer_map_dict[region]
            layer_to_regions.setdefault(layer, []).append(region)

        neuroid_layers = assembly['layer'].values
        neuroid_regions = np.array([
            '|'.join(layer_to_regions.get(layer, []))
            for layer in neuroid_layers
        ])
        return assembly.assign_coords(region=('neuroid', neuroid_regions))

    def start_recording(self,
                        recording_target: Union[str, List[str]],
                        time_bins: Optional[List[Tuple[int, int]]] = None,
                        recording_type: Optional[str] = None) -> None:
        if isinstance(recording_target, str):
            # Backward-compatible single-region path. Unknown strings pass
            # through as raw layer paths (preserves existing behavior where
            # `start_recording('layer3.2')` works without a region map entry).
            self._recording_layer = self._region_layer_map_dict.get(
                recording_target, recording_target
            )
            self._recording_regions = (
                [recording_target]
                if recording_target in self._region_layer_map_dict
                else []
            )
            self._recording_layers = [self._recording_layer]
            self._is_multi_region = False
        else:
            regions = list(recording_target)
            unknown = [r for r in regions
                       if r not in self._region_layer_map_dict]
            if unknown:
                raise ValueError(
                    f"Region(s) {unknown} not in region_layer_map "
                    f"(known regions: "
                    f"{list(self._region_layer_map_dict.keys())})"
                )
            self._recording_regions = regions
            # Deduplicate while preserving order — two regions can map to
            # the same layer; we extract that layer once and tag both.
            self._recording_layers = list(dict.fromkeys(
                self._region_layer_map_dict[r] for r in regions
            ))
            # Keep _recording_layer populated for code paths that still
            # read the singular attribute (e.g., legacy callable
            # preprocessors that take recording_layer=).
            self._recording_layer = (
                self._recording_layers[0]
                if len(self._recording_layers) == 1 else None
            )
            self._is_multi_region = len(regions) > 1
        self._time_bins = time_bins

    def start_task(self, task_context_or_task, fitting_stimuli=None,
                   **kwargs) -> None:
        if isinstance(task_context_or_task, TaskContext):
            self._task_context = task_context_or_task
        else:
            self._task_context = TaskContext(
                task_type=task_context_or_task,
                fitting_stimuli=fitting_stimuli,
            )

        task_type = self._task_context.task_type
        if not self._requires_behavioral_readout(task_type):
            # Passive / neural task — nothing to configure here
            self._use_generation_for_task = False
            self._readout_classifier = None
            return

        # Two possible behavioral paths depending on model capability and
        # what the TaskContext provides:
        #
        #   1. Instruction + generation_fn  → generation path (VLMs that
        #      natively do instruction-following lexical decision etc.)
        #   2. fitting_stimuli (no/either)  → readout path (feature models
        #      that need a trained logistic classifier)
        #
        # Both paths can be simultaneously supported. When the TaskContext
        # provides a ``prefer_path`` directive, that wins. When it does
        # not (or sets ``'auto'``), preference goes to generation when both
        # the model and TaskContext support it.
        instruction = self._task_context.instruction
        label_set = self._task_context.label_set
        fitting = self._task_context.fitting_stimuli
        prefer = self._task_context.prefer_path or 'auto'
        if prefer not in ('auto', 'generation', 'readout'):
            raise ValueError(
                f"TaskContext.prefer_path must be one of "
                f"('auto', 'generation', 'readout'); got {prefer!r}."
            )

        gen_viable = (self._generation_fn is not None
                      and instruction and label_set)
        readout_viable = fitting is not None

        if prefer == 'generation':
            if not gen_viable:
                raise ValueError(
                    f"Model '{self.identifier}' cannot run task "
                    f"'{task_type}' via generation path "
                    f"(prefer_path='generation'): "
                    f"requires model.generation_fn AND TaskContext.instruction "
                    f"AND TaskContext.label_set to be set."
                )
            self._use_generation_for_task = True
            self._readout_classifier = None
            return

        if prefer == 'readout':
            if not readout_viable:
                raise ValueError(
                    f"Model '{self.identifier}' cannot run task "
                    f"'{task_type}' via readout path "
                    f"(prefer_path='readout'): "
                    f"requires TaskContext.fitting_stimuli to be set."
                )
            self._use_generation_for_task = False
            self._fit_behavioral_readout(fitting)
            return

        # prefer == 'auto' — original dispatch precedence
        if gen_viable:
            self._use_generation_for_task = True
            self._readout_classifier = None
            return

        self._use_generation_for_task = False
        if readout_viable:
            self._fit_behavioral_readout(fitting)
        else:
            # No viable path — emit a clear error so the benchmark knows
            # this model cannot run the task.
            raise ValueError(
                f"Model '{self.identifier}' cannot run behavioral task "
                f"'{task_type}': TaskContext provides neither fitting_stimuli "
                f"(needed for readout path) nor (instruction + label_set "
                f"+ model.generation_fn) (needed for generation path)."
            )

    def reset(self) -> None:
        self._recording_layer = None
        self._recording_layers = []
        self._recording_regions = []
        self._is_multi_region = False
        self._task_context = None
        self._readout_classifier = None
        self._use_generation_for_task = False
        # Remove any active perturbations installed via process(StateChange).
        # Cleanup callables are tolerated to fail individually — we still
        # try to clear all of them so the model is left in a consistent state.
        for handle_id, cleanup in list(self._active_perturbations.items()):
            try:
                cleanup()
            except Exception:
                pass
        self._active_perturbations.clear()

    # ── Perturbation / state-change dispatch ──────────────────

    def _dispatch_state_change(self, state_change: 'StateChange') -> Any:
        """Route a StateChange event to the registered ``state_change_fn``,
        track the resulting cleanup callable, and return the
        ``PerturbationApplied`` confirmation.

        Two kinds of StateChange are handled here directly:

        - ``kind='reset'`` with a ``handle_id``: undo a single previously-
          applied perturbation. Useful when a benchmark wants to compose
          multiple perturbations independently.
        - Anything else: dispatch to ``state_change_fn``, which is expected
          to return ``(PerturbationApplied, cleanup_callable)``.
        """
        if state_change.kind == 'reset':
            handle_id = state_change.handle_id
            if handle_id is None:
                raise ValueError(
                    f"StateChange(kind='reset') requires handle_id to identify "
                    f"which perturbation to undo. Use model.reset() to clear all."
                )
            cleanup = self._active_perturbations.pop(handle_id, None)
            if cleanup is None:
                raise KeyError(
                    f"No active perturbation with handle_id={handle_id!r}. "
                    f"Active: {list(self._active_perturbations.keys())}."
                )
            cleanup()
            return None

        if self._state_change_fn is None:
            raise NotImplementedError(
                f"Model '{self.identifier}' has no state_change_fn registered. "
                f"Perturbation evaluation requires the model to declare a "
                f"state_change_fn(state_change) -> (PerturbationApplied, "
                f"cleanup) callable at BrainScoreModel construction time. "
                f"Received: StateChange(kind={state_change.kind!r})."
            )

        result = self._state_change_fn(state_change)
        if (not isinstance(result, tuple) or len(result) != 2
                or not isinstance(result[0], PerturbationApplied)
                or not callable(result[1])):
            raise TypeError(
                f"Model '{self.identifier}' state_change_fn must return a "
                f"(PerturbationApplied, cleanup_callable) tuple. Got "
                f"{type(result).__name__}."
            )
        applied, cleanup = result
        if applied.handle_id in self._active_perturbations:
            # Defensive: collision means state_change_fn issued a duplicate id
            raise ValueError(
                f"Duplicate handle_id {applied.handle_id!r} from state_change_fn. "
                f"state_change_fn must produce unique handle_ids per call."
            )
        self._active_perturbations[applied.handle_id] = cleanup
        return applied

    # ── Behavioral readout ────────────────────────────────────

    BEHAVIORAL_TASK_TYPES = frozenset({
        'probabilities', 'classification', 'label',
    })

    @classmethod
    def _requires_behavioral_readout(cls, task_type: str) -> bool:
        return task_type in cls.BEHAVIORAL_TASK_TYPES

    def _fit_behavioral_readout(self, fitting_stimuli) -> None:
        """Extract features at behavioral_readout_layer and fit a probabilities
        classifier.

        fitting_stimuli must carry labels. Convention (matches vision's
        ProbabilitiesMapping):
        - 'image_label' column for vision benchmarks
        - 'label' column as a generic fallback
        """
        from brainscore_core.behavior import ProbabilitiesClassifier

        if self._behavioral_readout_layer is None:
            raise ValueError(
                f"Model '{self.identifier}' was asked to perform a behavioral "
                f"task (task_type={self._task_context.task_type!r}) but no "
                f"behavioral_readout_layer is set. Register the model with "
                f"BrainScoreModel(..., behavioral_readout_layer='<layer name>')."
            )

        labels = self._extract_labels(fitting_stimuli)
        features = self._extract_behavioral_features(fitting_stimuli)

        self._readout_classifier = ProbabilitiesClassifier()
        self._readout_classifier.fit(features, labels)

    @staticmethod
    def _extract_labels(stimuli):
        for col in ('image_label', 'label'):
            if col in stimuli.columns:
                return list(stimuli[col].values)
        raise ValueError(
            f"Cannot find labels in fitting_stimuli. Expected one of "
            f"'image_label' or 'label' columns; got: {list(stimuli.columns)}."
        )

    def _extract_behavioral_features(self, stimuli):
        """Run the model once at the behavioral readout layer and return a
        2-D (presentation, neuroid) features assembly.

        Temporarily swaps _recording_layer so the existing process() path
        handles extraction unchanged.
        """
        saved_layer = self._recording_layer
        saved_layers = self._recording_layers
        saved_regions = self._recording_regions
        saved_multi = self._is_multi_region
        saved_classifier = self._readout_classifier
        self._recording_layer = self._behavioral_readout_layer
        self._recording_layers = (
            [self._behavioral_readout_layer]
            if self._behavioral_readout_layer else []
        )
        self._recording_regions = []
        self._is_multi_region = False
        # Bypass the behavioral predict path while fitting so we get features
        # not probabilities (matters when this helper is called from
        # _predict_probabilities for test stimuli too).
        self._readout_classifier = None
        try:
            features = self.process(stimuli)
        finally:
            self._recording_layer = saved_layer
            self._recording_layers = saved_layers
            self._recording_regions = saved_regions
            self._is_multi_region = saved_multi
            self._readout_classifier = saved_classifier

        # Ensure (presentation, neuroid) order
        if 'presentation' in features.dims and 'neuroid' in features.dims:
            features = features.transpose('presentation', 'neuroid')
        return features

    def _predict_probabilities(self, stimuli):
        """Return a BehavioralAssembly of per-label probabilities."""
        features = self._extract_behavioral_features(stimuli)
        return self._readout_classifier.predict_proba(features)

    def _generate_predictions(self, stimuli):
        """Call generation_fn per-stimulus, return one-hot BehavioralAssembly.

        Produces a (n_stimuli, n_labels) assembly where each row has a 1.0
        at the predicted label's position (and 0.0 elsewhere). This lets
        downstream argmax recover the predicted label the same way readout
        does, and it lets metrics that expect probability distributions
        still work (they just see a degenerate one-hot distribution).

        Invalid / unparseable responses default to the first label in
        label_set with a warning; the generation_fn is expected to enforce
        its own label discipline.
        """
        import numpy as np
        import pandas as pd
        from brainscore_core.supported_data_standards.brainio.assemblies import (
            BehavioralAssembly, walk_coords, array_is_element,
        )

        label_set = list(self._task_context.label_set)
        instruction = self._task_context.instruction
        label_to_idx = {lbl: i for i, lbl in enumerate(label_set)}

        n_stimuli = len(stimuli)
        n_labels = len(label_set)
        proba = np.zeros((n_stimuli, n_labels), dtype=np.float32)

        for i, (_, row) in enumerate(stimuli.iterrows()):
            predicted = self._generation_fn(
                stimulus_row=row,
                instruction=instruction,
                label_set=label_set,
            )
            if predicted not in label_to_idx:
                # Fall back to first label — generation_fn should handle
                # its own parsing robustness.
                import warnings
                warnings.warn(
                    f"generation_fn returned {predicted!r}, not in "
                    f"label_set={label_set}. Defaulting to {label_set[0]!r}."
                )
                predicted = label_set[0]
            proba[i, label_to_idx[predicted]] = 1.0

        # Build presentation coords from the stimulus set
        stimulus_ids = list(stimuli['stimulus_id'].values)
        presentation_coords = {
            'stimulus_id': ('presentation', stimulus_ids),
        }
        for column in stimuli.columns:
            if column == 'stimulus_id':
                continue
            presentation_coords[column] = ('presentation', list(stimuli[column].values))

        return BehavioralAssembly(
            proba,
            coords={
                **presentation_coords,
                'choice': label_set,
            },
            dims=['presentation', 'choice'],
        )

    # -- Legacy compatibility methods --
    # These allow BrainScoreModel to be used by existing vision and language
    # benchmarks that call look_at(), digest_text(), etc. They are NOT part
    # of the UnifiedModel ABC -- they exist only on BrainScoreModel to bridge
    # the gap until benchmarks migrate to process().

    def look_at(self, stimuli, number_of_trials=1, **kwargs):
        """Vision benchmark compatibility. Delegates to process()."""
        return self.process(stimuli)

    def visual_degrees(self) -> int:
        """Vision benchmark compatibility."""
        return self._visual_degrees_val

    def digest_text(self, text) -> Dict[str, Any]:
        """Language benchmark compatibility. Wraps process() result in
        the expected {'neural': ..., 'behavior': ...} dict.

        Stimulus identifier includes a hash of the text content so repeated
        calls with different sentences do not collide in the @store_xarray
        cache on the activations_model.
        """
        import hashlib
        import pandas as pd
        from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet

        import numpy as np
        if isinstance(text, np.ndarray):
            text = text.tolist()
        if isinstance(text, (str, list)):
            if isinstance(text, str):
                text = [text]
            stimuli = StimulusSet(pd.DataFrame({
                'sentence': text,
                'stimulus_id': list(range(len(text))),
            }))
            content_hash = hashlib.md5(
                '|'.join(text).encode('utf-8')).hexdigest()[:12]
            stimuli.identifier = f'text_stimuli_{len(text)}_{content_hash}'
        else:
            stimuli = text

        result = self.process(stimuli)
        return {'neural': result}

    def start_neural_recording(self, recording_target, recording_type='fMRI'):
        """Language benchmark compatibility."""
        self.start_recording(recording_target, recording_type=recording_type)
