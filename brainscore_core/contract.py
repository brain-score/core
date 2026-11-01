"""Core Brain-Score subject contract."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from .events import InputEvent, OutputEvent
from .io_catalog import modalities_to_input_channels

if TYPE_CHECKING:
    from .streaming import Session
    from .selection import UnitSelector


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



class Subject(ABC):
    """Minimal contract for a subject evaluated through a session.

    Declare identity and supported channels, then implement ``interact`` to
    consume inputs with ``session.next_input()`` and submit responses with
    ``session.respond(event)``. A subject need not expose layers, modalities, or
    legacy model methods. Declarations may be properties or class attributes.

    ``reset`` is a lifecycle hook, not an evaluation method. Stateful subjects
    override it to clear memory and interventions between independent runs.
    """

    @property
    @abstractmethod
    def identifier(self) -> str:
        """Stable subject name."""
        ...

    @property
    @abstractmethod
    def in_channels(self) -> Set[str]:
        """Input channels the subject can consume."""
        ...

    @property
    @abstractmethod
    def out_channels(self) -> Set[str]:
        """Output channels the subject can emit."""
        ...

    @property
    def required_channels(self) -> Set[str]:
        """Inputs the subject cannot run without; none by default."""
        return set()

    @abstractmethod
    def interact(self, session: "Session") -> None:
        """Consume session inputs and emit responses until the session ends."""
        ...

    def reset(self) -> None:
        """Clear state between independent runs; stateless subjects do nothing."""
        pass


class UnifiedModel(Subject):
    """Subject base for models evaluated through typed paths.

    Implementations provide ``process``, ``start_task``, ``start_recording``,
    a layer map and modality declarations. ``interact`` (the evaluation
    method of the contract) runs a session through those typed paths, so a
    typed model needs no session code of its own. ``process`` is a typed
    convenience of this base, not part of the subject contract.

    New session-native implementations should inherit ``Subject`` directly.
    """

    @property
    @abstractmethod
    def identifier(self) -> str:
        ...

    @property
    @abstractmethod
    def region_layer_map(self) -> Mapping[str, Union[str, "UnitSelector"]]:
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

    @property
    def in_channels(self) -> Set[str]:
        """Input channels: the declared modalities as channel families."""
        return modalities_to_input_channels(self.available_modalities)

    @property
    def out_channels(self) -> Set[str]:
        """Output channels: ``neural:<region>`` for every mapped region.

        Typed models that also answer tasks or act override this to add
        ``behavior`` or ``motor``; nothing is discovered from attribute names.
        """
        return {f"neural:{region}" for region in self.region_layer_map}

    @property
    def required_channels(self) -> Set[str]:
        """Input stream channels this subject hard-requires."""
        return modalities_to_input_channels(self.required_modalities)

    @abstractmethod
    def process(self, input_event: InputEvent) -> OutputEvent:
        ...

    def interact(self, session: "Session") -> None:
        """Run ``session`` through this model's typed paths."""
        from .streaming_helpers import drive_typed_session
        drive_typed_session(self, session)

    def start_task(self, task_context: TaskContext) -> None:
        self._task_context: Optional[TaskContext] = task_context

    def start_recording(self,
                        recording_target: Union[str, List[str]],
                        time_bins: Optional[List[Tuple[int, int]]] = None,
                        recording_type: Optional[str] = None) -> None:
        pass

    def reset(self) -> None:
        pass
