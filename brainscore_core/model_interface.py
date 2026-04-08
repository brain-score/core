"""
Unified model interface for Brain-Score.

Defines the core abstractions for model evaluation:
- TaskContext: benchmark-to-model communication
- UnifiedModel: the single evaluation interface (ABC)
- ModalityProcessor: per-modality processing strategy (ABC)
- ModalityIntegrator: cross-modal fusion (ABC)
- BrainScoreModel: compositional implementation with three-case dispatch
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class TaskContext:
    """
    Everything a model needs to understand what task the benchmark wants.

    Benchmarks populate all relevant fields. Models consume whichever
    fields match their paradigm:
    - Feature models use fitting_stimuli + label_set to train a readout
    - Instruction-following models use instruction + label_set to build a prompt
    - Models that support both can choose their preferred approach
    """
    task_type: str
    label_set: Optional[List[str]] = None
    fitting_stimuli: Optional[Any] = None
    instruction: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedModel(ABC):
    """
    Base interface for all Brain-Score models.

    Defines identity, lifecycle, and a single processing method.
    Benchmarks call process(stimuli) for all evaluation. What the model
    perceives is determined by the stimulus content. What the model
    produces is determined by the measurement configuration (start_task
    for behavioral, start_recording for neural).
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
        ...

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


class ModalityProcessor(ABC):
    """
    Handles input preprocessing, model invocation, and output extraction
    for one modality. Processors are stateless -- BrainScoreModel manages
    recording targets, task context, and other state.
    """

    @property
    @abstractmethod
    def modality(self) -> str:
        ...

    @abstractmethod
    def __call__(self, model, stimuli, *,
                 recording_layer: Optional[str] = None,
                 task_context: Optional[TaskContext] = None,
                 **kwargs) -> Any:
        ...


class ModalityIntegrator(ABC):
    """
    Combines features extracted by multiple ModalityProcessors into a
    single output. Used by models with explicit cross-modal fusion
    (e.g., a cross-attention module that takes vision + text features).
    """

    @abstractmethod
    def integrate(self, modality_features: Dict[str, Any], *,
                  recording_layer: Optional[str] = None,
                  task_context: Optional[TaskContext] = None) -> Any:
        ...


class BrainScoreModel(UnifiedModel):
    """
    Compositional implementation of UnifiedModel.

    Accepts modality processors that handle input/output for common model
    types. process() dispatches to the appropriate processor(s) based on
    stimulus content via three cases:

    Case 1: Single modality -- dispatch to that processor.
    Case 2: Multiple modalities + integrator -- extract per-modality, then fuse.
    Case 3: Multiple modalities + primary_processor (VLMs) -- dispatch to primary.
    """

    COLUMN_TO_MODALITY: Dict[str, str] = {
        'image_file_name': 'vision',
        'image_path': 'vision',
        'filename': 'vision',
        'sentence': 'text',
        'text': 'text',
        'video_path': 'video',
        'audio_path': 'audio',
    }

    def __init__(
        self,
        identifier: str,
        model: Any,
        region_layer_map: Dict[str, str],
        processors: List[ModalityProcessor],
        integrator: Optional[ModalityIntegrator] = None,
        primary_processor: Optional[str] = None,
    ) -> None:
        self._identifier_str = identifier
        self._model = model
        self._region_layer_map_dict = region_layer_map
        self._processors: Dict[str, ModalityProcessor] = {
            p.modality: p for p in processors
        }
        self._integrator = integrator
        self._primary_processor = primary_processor
        self._recording_layer: Optional[str] = None
        self._task_context: Optional[TaskContext] = None

        if primary_processor and primary_processor not in self._processors:
            raise ValueError(
                f"primary_processor '{primary_processor}' does not match any "
                f"registered processor modality. Available: "
                f"{set(self._processors.keys())}"
            )
        if primary_processor and integrator:
            raise ValueError(
                "Cannot specify both primary_processor and integrator. "
                "Use primary_processor for VLMs that handle multimodal input "
                "in a single forward pass. Use integrator for models with an "
                "explicit cross-modal fusion stage."
            )

    @property
    def identifier(self) -> str:
        return self._identifier_str

    @property
    def region_layer_map(self) -> Dict[str, str]:
        return dict(self._region_layer_map_dict)

    @property
    def supported_modalities(self) -> Set[str]:
        return set(self._processors.keys())

    def _detect_modalities(self, stimuli) -> Set[str]:
        detected: Set[str] = set()
        for col in stimuli.columns:
            modality = self.COLUMN_TO_MODALITY.get(col)
            if modality and modality in self._processors:
                detected.add(modality)
        return detected

    def process(self, stimuli) -> Any:
        detected = self._detect_modalities(stimuli)

        if not detected:
            raise ValueError(
                f"No recognized modality columns in stimulus set. "
                f"Columns present: {list(stimuli.columns)}. "
                f"Known column mappings: {self.COLUMN_TO_MODALITY}. "
                f"Model supports: {self.supported_modalities}."
            )

        # Case 1: Single modality
        if len(detected) == 1:
            modality = next(iter(detected))
            return self._processors[modality](
                self._model, stimuli,
                recording_layer=self._recording_layer,
                task_context=self._task_context,
            )

        # Case 2: Multiple modalities + integrator
        if self._integrator is not None:
            modality_features: Dict[str, Any] = {}
            for modality in detected:
                modality_features[modality] = self._processors[modality](
                    self._model, stimuli,
                    recording_layer=None,
                    task_context=self._task_context,
                )
            return self._integrator.integrate(
                modality_features,
                recording_layer=self._recording_layer,
                task_context=self._task_context,
            )

        # Case 3: Multiple modalities, no integrator (VLM)
        if self._primary_processor is not None:
            return self._processors[self._primary_processor](
                self._model, stimuli,
                recording_layer=self._recording_layer,
                task_context=self._task_context,
            )

        raise ValueError(
            f"Multiple modalities detected in stimuli ({detected}) but "
            f"model has neither an integrator nor a primary_processor "
            f"designated. For VLMs that handle multimodal input in a "
            f"single forward pass, set primary_processor to the modality "
            f"of the processor that should receive the full stimulus set. "
            f"For models with explicit cross-modal fusion, provide an "
            f"integrator."
        )

    def start_recording(self, recording_target: str,
                        time_bins: Optional[List[Tuple[int, int]]] = None,
                        recording_type: Optional[str] = None) -> None:
        self._recording_layer = self._region_layer_map_dict.get(
            recording_target, recording_target
        )
        self._time_bins = time_bins

    def start_task(self, task_context: TaskContext) -> None:
        self._task_context = task_context

    def reset(self) -> None:
        self._recording_layer = None
        self._task_context = None
