"""
Unified model interface for Brain-Score.

Defines the core abstractions for model evaluation:
- TaskContext: benchmark-to-model communication
- UnifiedModel: the single evaluation interface (ABC)
- BrainScoreModel: compositional implementation with preprocessors + shared activations_model
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


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
    }

    def __init__(
        self,
        identifier: str,
        model: Any,
        region_layer_map: Dict[str, str],
        preprocessors: Dict[str, Callable],
        activations_model: Any = None,
        visual_degrees: int = 8,
    ) -> None:
        self._identifier_str = identifier
        self._model = model
        self._region_layer_map_dict = region_layer_map
        self._preprocessors = preprocessors
        self._activations_model = activations_model
        self._visual_degrees_val = visual_degrees
        self._recording_layer: Optional[str] = None
        self._task_context: Optional[TaskContext] = None

    @property
    def identifier(self) -> str:
        return self._identifier_str

    @property
    def region_layer_map(self) -> Dict[str, str]:
        return dict(self._region_layer_map_dict)

    @property
    def supported_modalities(self) -> Set[str]:
        return set(self._preprocessors.keys())

    def _detect_modalities(self, stimuli) -> Set[str]:
        detected: Set[str] = set()
        for col in stimuli.columns:
            modality = self.COLUMN_TO_MODALITY.get(col)
            if modality and modality in self._preprocessors:
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

        modality = next(iter(detected))

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

    def reset(self) -> None:
        self._recording_layer = None
        self._task_context = None

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
        the expected {'neural': ..., 'behavior': ...} dict."""
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
            stimuli.identifier = 'text_stimuli'
        else:
            stimuli = text

        result = self.process(stimuli)
        return {'neural': result}

    def start_neural_recording(self, recording_target, recording_type='fMRI'):
        """Language benchmark compatibility."""
        self.start_recording(recording_target, recording_type=recording_type)
