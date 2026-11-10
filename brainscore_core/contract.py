"""Core Brain-Score subject contract."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from .events import InputEvent, OutputEvent
from .io_catalog import modalities_to_input_channels

if TYPE_CHECKING:
    from .streaming import Session


# How a session groups its trials into episodes (begin ... end). ``independent``:
# every trial is its own episode, so a subject keeps nothing from earlier trials.
# ``conversation``: all trials are one episode, so a subject with the
# ``memory:trial_history`` trait sees its earlier trials and responses.
TRIAL_PROTOCOLS = ('independent', 'conversation')
PROTOCOL_REQUIRED_TRAITS = {'independent': frozenset(), 'conversation': frozenset({'memory:trial_history'})}


def required_traits_for(declared, trial_protocol: str = 'independent') -> Set[str]:
    """The traits a subject needs: the ``declared`` ones plus those the protocol implies.

    The one place that derives a required trait from a trial protocol, so a
    benchmark author names the protocol and never the trait.
    """
    if trial_protocol not in TRIAL_PROTOCOLS:
        raise ValueError(f'trial_protocol must be one of {TRIAL_PROTOCOLS}; got {trial_protocol!r}')
    return set(declared or ()) | PROTOCOL_REQUIRED_TRAITS[trial_protocol]


@dataclass
class TaskContext:
    """
    Everything a model needs to understand what task the benchmark wants.

    Benchmarks populate all relevant fields. Models consume whichever
    fields match their paradigm:
    - Feature models use fitting_stimuli + label_set to train a readout
      (``BrainScoreModel(behavioral_readout_layer=...)``)
    - Instruction-following models use instruction + label_set with their
      ``answer`` function (``BrainScoreModel(answer=...)``)
    - Models that support both can choose their preferred approach

    ``prefer_path`` lets a benchmark (or an experimenting user) pin which
    behavioral dispatch path is selected when a model supports more than
    one. Accepts ``'auto'`` (default — generation if available, else
    readout), ``'generation'`` (force generation through the model's
    ``answer``; error if no answer is registered or instruction/label_set
    is empty), or ``'readout'`` (force readout, error if
    behavioral_readout_layer is missing or fitting_stimuli is empty). Use
    it instead of disabling a model's answer function to force readout.

    ``BrainScoreModel`` accepts ``task_type`` ``'passive'`` or one of
    ``'probabilities'``, ``'classification'``, ``'label'``. Put the stimuli to
    answer (for example a held-out test set) in ``metadata['stimulus_set']``
    when using ``behavioral_response``.

    ``response`` is the form of the answer (``brainscore_core.responses``):
    ``Choice``, ``Scale`` or ``Text``; ``task_type='passive'`` means no answer.
    With a response, outputs are requested on ``behavior:<kind>`` channels.
    The older ``task_type`` values ``'label'``, ``'classification'`` and
    ``'probabilities'`` keep their meaning on the plain ``behavior`` channel.

    ``present`` gives the order of a trial's parts when a stimulus has several
    (e.g. ``('text', 'video')``: a sentence, then a clip). Name columns to show
    several parts of one kind in sequence: ``('image:instruction', 'image:word_1',
    'image:word_2')``. A replying model is
    shown the instruction, those parts in order, then the response's directive.

    ``trial_protocol`` says how the session groups trials into episodes:
    ``'independent'`` (default) wraps each trial (one stimulus row and its
    response) in ``begin`` ... ``end('terminal')``, so a subject starts every
    trial with empty history; ``'conversation'`` sends one ``begin``, all trials,
    one ``end('terminal')``, so a subject with the ``memory:trial_history``
    trait sees its earlier trials. A benchmark that uses ``'conversation'``
    also sets the same ``trial_protocol`` attribute on itself; pre-flight
    derives the required trait from it (``required_traits_for``).

    ``readout_fit`` says which network a readout is fitted on while a change
    (a lesion, a stimulation) is active: ``'current'`` (default) fits on the
    changed network; ``'intact'`` fits on the network with every change undone,
    then tests the changed network with that readout, as an animal trained
    before a lesion is tested after it.
    """
    task_type: Optional[str] = None
    label_set: Optional[List[str]] = None
    fitting_stimuli: Optional[Any] = None
    instruction: Optional[str] = None
    prefer_path: Optional[str] = None  # 'auto' | 'generation' | 'readout'
    metadata: Dict[str, Any] = field(default_factory=dict)
    readout_fit: str = 'current'  # 'current' | 'intact'
    response: Any = None  # a brainscore_core.responses Choice, Scale or Text
    present: Optional[Tuple[str, ...]] = None  # order of a trial's parts: 'text', 'image', 'video', 'audio'
    trial_protocol: str = 'independent'  # 'independent' | 'conversation'

    def __post_init__(self):
        if self.trial_protocol not in TRIAL_PROTOCOLS:
            raise ValueError(f'trial_protocol must be one of {TRIAL_PROTOCOLS}; got {self.trial_protocol!r}')
        # The response, when given, decides the task type; label lists follow it.
        kinds = {'Choice': 'choice', 'Scale': 'rating', 'Text': 'text'}
        if self.response is not None:
            kind = kinds.get(type(self.response).__name__)
            if kind is None:
                raise TypeError('response must be a Choice, Scale or Text from brainscore_core.responses')
            if self.task_type is None:
                self.task_type = kind
            elif self.task_type != kind:
                raise ValueError(f'task_type {self.task_type!r} contradicts a {type(self.response).__name__} response')
            if kind == 'choice' and self.label_set is None:
                self.label_set = list(self.response.labels)
        elif self.task_type is None:
            raise ValueError("Give a response (Choice, Scale, Text) or task_type='passive'.")
        if self.present is not None:
            kinds = {'text', 'image', 'video', 'audio'}
            unknown = [entry for entry in self.present if entry.split(':', 1)[0] not in kinds]
            if unknown or isinstance(self.present, str):
                raise ValueError("present lists parts as a kind ('text', 'image', 'video', 'audio') "
                                 f"or 'kind:column' (e.g. 'image:word_1'); got {self.present!r}")
            self.present = tuple(self.present)



class Subject(ABC):
    """Minimal contract for a subject evaluated through a session.

    Declare identity and supported channels, then implement ``participate`` to
    consume inputs with ``session.next_input()`` and submit responses with
    ``session.respond(event)``. A subject need not expose layers, modalities, or
    legacy model methods. Declarations may be properties or class attributes.

    ``reset`` is a lifecycle hook, not an evaluation method. Stateful subjects
    override it to clear memory, task setup and interventions between independent
    runs. A ``begin`` control event is narrower: it clears only the history ``h``
    (earlier trials, steps and responses, and the episode state of the subject's
    parts); interventions and task setup stay.
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

    @property
    def traits(self) -> Set[str]:
        """Properties of the subject a benchmark may require; none by default.

        ``'memory:trial_history'``: the subject keeps earlier trials, earlier
        steps and its responses from ``begin`` until ``end``. A subject without
        it always starts a trial with empty history. A benchmark that needs a
        trait lists it in ``required_traits`` (a ``conversation`` trial protocol
        implies this one); a subject without it is N/A.
        """
        return set()

    @abstractmethod
    def participate(self, session: "Session") -> None:
        """Consume session inputs and emit responses until the session ends."""
        ...

    def reset(self) -> None:
        """Clear state between independent runs; stateless subjects do nothing."""
        pass


class UnifiedModel(Subject):
    """Subject base for models evaluated through typed paths.

    Implementations provide ``process``, ``start_task``, ``start_recording``,
    a layer map and modality declarations. ``participate`` (the evaluation
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
    def region_layer_map(self) -> Mapping[str, Any]:
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

    def process(self, input_event: InputEvent) -> OutputEvent:
        """Typed convenience used by the default ``participate``; not part of the contract."""
        raise NotImplementedError(
            f"{type(self).__name__} defines no process(); implement participate(session) "
            "or process(input_event) for the typed session driver.")

    def participate(self, session: "Session") -> None:
        """Run ``session`` through this model's typed paths."""
        from brainscore_core._sessions.helpers import drive_typed_session
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
