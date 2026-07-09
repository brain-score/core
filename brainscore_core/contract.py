"""Core Brain-Score subject contract."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .events import InputEvent, OutputEvent
from .io_catalog import modalities_to_input_channels


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
    """
    Base interface for all Brain-Score subjects.

    Renamed from ``UnifiedModel`` in v1.5. The interface describes a *subject*,
    not specifically a model: every event and measurement is something you could
    present to, or read from, a biological subject through a human harness, not
    only a model. ``UnifiedModel`` remains a deprecated alias (defined below the
    class) for one release so existing imports keep working unchanged.

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

    @property
    def in_channels(self) -> Set[str]:
        """Input stream channels this subject can consume.

        Defaults to the v1.5 modality declarations so existing subjects get a
        conservative v2 channel identity without overriding this property.
        """
        return self._modalities_to_input_channels(
            self._declared_input_modalities()
        )

    @property
    def out_channels(self) -> Set[str]:
        """Output stream channels this subject can emit."""
        channels = {
            f"neural:{region}"
            for region in self.region_layer_map
        }
        if self._has_behavioral_output_path():
            channels.add("behavior")
        if self._has_action_output_path():
            channels.add("motor")
        return channels

    @property
    def required_channels(self) -> Set[str]:
        """Input stream channels this subject hard-requires."""
        return self._modalities_to_input_channels(self.required_modalities)

    @abstractmethod
    def process(self, input_event: InputEvent) -> OutputEvent:
        ...

    def interact(self, session) -> None:
        identifier = self._safe_identifier()
        raise NotImplementedError(
            f"Subject {identifier} has no v2 interact() path yet; use "
            f"process()/start_recording()."
        )

    def start_task(self, task_context: TaskContext) -> None:
        self._task_context: Optional[TaskContext] = task_context

    def start_recording(self,
                        recording_target: Union[str, List[str]],
                        time_bins: Optional[List[Tuple[int, int]]] = None,
                        recording_type: Optional[str] = None) -> None:
        pass

    def reset(self) -> None:
        pass

    def _declared_input_modalities(self) -> Set[str]:
        for attr in ("available_modalities", "supported_modalities"):
            try:
                modalities = getattr(self, attr)
            except (AttributeError, NotImplementedError):
                continue
            if modalities is not None:
                return set(modalities)

        preprocessors = getattr(self, "_preprocessors", None)
        if hasattr(preprocessors, "keys"):
            return set(preprocessors.keys())
        return set()

    @staticmethod
    def _modalities_to_input_channels(modalities) -> Set[str]:
        return modalities_to_input_channels(modalities)

    def _has_behavioral_output_path(self) -> bool:
        for attr in (
            "_behavioral_readout_layer",
            "_generation_fn",
            "behavioral_readout_layer",
            "generation_fn",
        ):
            try:
                value = getattr(self, attr)
            except Exception:
                continue
            if value is not None:
                return True
        return False

    def _has_action_output_path(self) -> bool:
        for attr in ("_action_fn", "action_fn"):
            try:
                value = getattr(self, attr)
            except Exception:
                continue
            if value is not None:
                return True
        return False

    def _safe_identifier(self) -> str:
        try:
            return str(self.identifier)
        except Exception:
            return type(self).__name__


# Deprecated alias. ``UnifiedModel`` was the v1 name for the subject contract.
# Renamed to ``Subject`` in v1.5 to make the interface subject-agnostic: a model
# or a future human harness satisfies the same contract. Kept as an alias for one
# release so existing imports (``from brainscore_core import UnifiedModel``) keep
# working. New code should use ``Subject``.
UnifiedModel = Subject
