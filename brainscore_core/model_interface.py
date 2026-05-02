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
class StateChange:
    """Induce a state change in the model (dysfunction, lesion, perturbation).

    Used to simulate conditions like dyslexia, prosopagnosia, or pharmacological
    effects. Passed to model.process() the same way stimuli are — the interface
    is generic over input event type so benchmarks studying neural dysfunction
    or drug effects do not need a separate model method.

    NOT YET IMPLEMENTED in BrainScoreModel (reserved for future work). Concrete
    models that support state changes override process() or subclass
    BrainScoreModel to dispatch on input type.

    Examples (conceptual — not yet functional):
        StateChange('dyslexia')
        StateChange('lesion', {'region': 'V4'})
        StateChange('pharmacological', {'drug': 'propofol', 'dose': 0.5})
    """
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)


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

    def start_recording(self, recording_target: str,
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
        self._recording_layer: Optional[str] = None
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
        # for embodied evaluation. StateChange remains reserved for future
        # work — see the docstring on that class.
        if isinstance(input_event, StateChange):
            raise NotImplementedError(
                f"State changes are not yet implemented on BrainScoreModel. "
                f"Received: StateChange(kind={input_event.kind!r}). "
                f"Concrete models that support dysfunction/lesion/perturbation "
                f"should subclass BrainScoreModel and override process()."
            )
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

        # Vision path: delegate to activations_model (PytorchWrapper handles
        # preprocessing, forward pass, hooks, caching, and assembly packaging)
        if modality == 'vision' and self._activations_model is not None:
            layers = [self._recording_layer] if self._recording_layer else []
            return self._activations_model(stimuli, layers=layers)

        # Other modalities: check if the preprocessor is a full extractor
        # (like TextWrapper — accepts (stimuli, layers=[])) or a legacy
        # callable (accepts (model, stimuli, recording_layer=...)).
        # Duck-type: extractors have an `identifier` attribute.
        preprocessor = self._preprocessors[modality]
        layers = [self._recording_layer] if self._recording_layer else []
        if hasattr(preprocessor, 'identifier'):
            return preprocessor(stimuli, layers=layers)
        return preprocessor(
            self._model, stimuli,
            recording_layer=self._recording_layer,
        )

    def start_recording(self, recording_target: str,
                        time_bins: Optional[List[Tuple[int, int]]] = None,
                        recording_type: Optional[str] = None) -> None:
        self._recording_layer = self._region_layer_map_dict.get(
            recording_target, recording_target
        )
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
        self._task_context = None
        self._readout_classifier = None
        self._use_generation_for_task = False

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
        saved_classifier = self._readout_classifier
        self._recording_layer = self._behavioral_readout_layer
        # Bypass the behavioral predict path while fitting so we get features
        # not probabilities (matters when this helper is called from
        # _predict_probabilities for test stimuli too).
        self._readout_classifier = None
        try:
            features = self.process(stimuli)
        finally:
            self._recording_layer = saved_layer
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
