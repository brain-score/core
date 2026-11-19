"""Core Brain-Score subject contract."""

import functools
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Mapping, Optional, Set, Tuple


if TYPE_CHECKING:
    from .streaming import Session


# How a session groups its trials into episodes (begin ... end). ``none``:
# every trial is its own episode, so a subject keeps nothing from earlier trials.
# ``history``: all trials are one episode, so a subject with the
# ``memory:trial_history`` trait sees its earlier trials and responses.
_TRIAL_MEMORIES = ('none', 'history')
_MEMORY_REQUIRED_TRAITS = {'none': frozenset(), 'history': frozenset({'memory:trial_history'})}


def required_traits_for(declared, trial_memory: str = 'none') -> Set[str]:
    """The traits a subject needs: the ``declared`` ones plus those the trial memory implies.

    The one place that derives a required trait from a trial memory, so a
    benchmark author names the memory and never the trait.
    """
    if trial_memory not in _TRIAL_MEMORIES:
        raise ValueError(f'trial_memory must be one of {_TRIAL_MEMORIES}; got {trial_memory!r}')
    return set(declared or ()) | _MEMORY_REQUIRED_TRAITS[trial_memory]


@dataclass
class TaskContext:
    """
    Everything a model needs to understand what task the benchmark wants.

    Benchmarks populate all relevant fields. Models consume whichever
    fields match their paradigm:
    - A readout is fitted on ``fitting_stimuli`` with the labels of the
      response (``BrainScoreModel(behavioral_readout_layer=...)``)
    - An ``answer``, ``reply`` or ``likelihood`` part reads the instruction
      and the response's options
    - Which part answers each format is a property of the model, not of the
      task: the one part that gives the format, or the part that the
      registration states

    ``BrainScoreModel`` answers a task with a ``response`` or ``task_type='passive'``.
    Put the stimuli to answer (for example a held-out test set) to
    ``behavioral_response`` or ``stimulus_session`` as ``stimuli``.

    ``response`` is the form of the answer (``brainscore_core.responses``):
    ``Choice``, ``Scale`` or ``Text``; ``task_type='passive'`` means no answer.
    With a response, outputs are requested on ``behavior:<kind>`` channels.
    The older ``task_type`` values (``'label'``, ``'probabilities'``, ``'odd_one_out'``) are
    the calls of old benchmarks and old models; the legacy facade asks an old
    probabilities task as a ``Choice`` on ``behavior:probabilities``.

    ``present`` gives the order of a trial's parts when a stimulus has several
    (e.g. ``('text', 'video')``: a sentence, then a clip). Name columns to show
    several parts of one kind in sequence: ``('image:instruction', 'image:word_1',
    'image:word_2')``. A replying model is
    shown the instruction, those parts in order, then the response's directive.

    ``trial_memory`` says how long a subject keeps its trials:
    ``'none'`` (default) wraps each trial (one stimulus row and its
    response) in ``begin`` ... ``end('terminal')``, so a subject starts every
    trial with empty history; ``'history'`` sends one ``begin``, all trials,
    one ``end('terminal')``, so a subject with the ``memory:trial_history``
    trait sees its earlier trials. A benchmark that uses ``'history'``
    also sets the same ``trial_memory`` attribute on itself; pre-flight
    derives the required trait from it (``required_traits_for``).

    ``readout_fit`` says which network a readout is fitted on while a change
    (a lesion, a stimulation) is active: ``'current'`` (default) fits on the
    changed network; ``'intact'`` fits on the network with every change of the
    subject undone, then tests the changed network with that readout, as an
    animal trained before a lesion is tested after it. A change made outside the
    subject (a tool that hooks its network) is not undone for the fit.
    """
    task_type: Optional[str] = None
    label_set: Optional[List[str]] = None
    fitting_stimuli: Optional[Any] = None
    instruction: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    readout_fit: str = 'current'  # 'current' | 'intact'
    response: Any = None  # a brainscore_core.responses Choice, Scale, Text or Segments
    present: Optional[Tuple[str, ...]] = None  # order of a trial's parts: 'text', 'image', 'video', 'audio'
    trial_memory: str = 'none'  # 'none' | 'history'

    def __post_init__(self):
        if self.trial_memory not in _TRIAL_MEMORIES:
            raise ValueError(f'trial_memory must be one of {_TRIAL_MEMORIES}; got {self.trial_memory!r}')
        # The response, when given, decides the task type; label lists follow it.
        from .responses import Choice, Scale, Segments, Text
        # By identity: the response classes are the core formats, never a subclass.
        kinds = {Choice: 'choice', Scale: 'rating', Text: 'text', Segments: 'segments'}
        if self.response is not None:
            kind = kinds.get(type(self.response))
            if kind is None:
                raise TypeError('response must be a Choice, Scale, Text or Segments from '
                                'brainscore_core.responses (not a subclass)')
            if self.task_type is None:
                self.task_type = kind
            elif self.task_type != kind:
                raise ValueError(f'task_type {self.task_type!r} contradicts a {type(self.response).__name__} response')
            if (kind == 'choice' and self.label_set is None
                    and not isinstance(self.response.labels, str)):
                self.label_set = list(self.response.labels)
            if kind == 'choice' and self.response.verbatim and (self.instruction or '').strip():
                raise ValueError('A verbatim response shows the stimulus text exactly as given; '
                                 'remove the instruction or put it in the text.')
        elif self.task_type is None:
            raise ValueError("Give a response or set task_type='passive'.")
        if self.present is not None:
            kinds = {'text', 'image', 'video', 'audio'}
            unknown = [entry for entry in self.present if entry.split(':', 1)[0] not in kinds]
            if unknown or isinstance(self.present, str):
                raise ValueError("present lists parts as a kind ('text', 'image', 'video', 'audio') "
                                 f"or 'kind:column' (e.g. 'image:word_1'); got {self.present!r}")
            self.present = tuple(self.present)


def _take_part(subject, body, session):
    """Run ``body(session)`` with the session duties: claim the subject, declare its inputs,
    stage the session's notes until the participation returns."""
    from ._sessions import one_at_a_time
    from ._evaluation.measures import stage_session
    # A nested call (super().participate, a wrapper that delegates the same session) passes
    # while this thread's own claim for this session is live and its inputs are declared.
    if one_at_a_time.declared(subject, session):
        return _run_now(subject, body, session)
    with one_at_a_time.one_session_at_a_time(subject, session) as claim, stage_session(session, subject):
        session.declare_inputs(subject.in_channels)
        claim[3] = True
        return _run_now(subject, body, session)


def _run_now(subject, body, session):
    result = body(session)
    # A generator or coroutine body would run later, outside the claim.
    if inspect.isgenerator(result) or inspect.iscoroutine(result) or inspect.isasyncgen(result):
        getattr(result, 'close', lambda: None)()
        raise TypeError(f'{type(subject).__name__}.participate returned a {type(result).__name__}; '
                        'participate runs the whole session before it returns: make it a plain '
                        'method (no yield, no async def).')
    return result


class Subject(ABC):
    """Minimal contract for a subject evaluated through a session.

    Declare identity and supported channels, then implement ``participate`` to
    consume inputs with ``session.next_input()`` and submit responses with
    ``session.respond(event)``. A subject need not expose layers, modalities, or
    legacy model methods.

    Declare with class attributes, or with properties when a declaration depends
    on a constructor argument. ``self.out_channels = ...`` in ``__init__`` does
    not work: the abstract properties must be defined on the class::

        class Fixed(Subject):
            identifier = 'fixed'
            in_channels = {'text'}
            out_channels = {'behavior:choice'}

        class Configured(Subject):
            def __init__(self, outputs):
                self._outputs = set(outputs)

            identifier = 'configured'
            in_channels = {'text'}

            @property
            def out_channels(self):
                return self._outputs

    What runs around ``participate``: reading ``subject.participate`` gives the
    method that Python's normal lookup finds (the class body, a parent, a mixin,
    or a method assigned to the class or the instance later), with the session
    duties of the base around it. When it is called with a session, the base

    1. claims the subject for this session: a second session on the same subject
       object, from this thread or another one, is refused with ``RuntimeError``
       while the first one runs (the recording and task state belong to one
       session). Build a separate subject for each session that runs at the same
       time.
    2. calls ``session.declare_inputs(self.in_channels)``: an input on a channel
       the subject does not declare never reaches the body, and ``control``
       events reach it only when it declares ``control``.
    3. runs the body. The body must run the whole session before it returns: a
       ``participate`` that yields or is ``async`` is refused. A ``participate``
       that cannot be called (``participate = None``) is refused when it is read,
       so the compatibility check refuses it before any session.
    4. keeps what the session notes for the score (its delivery, responses,
       provenance) apart until the body returns: then it passes the notes on and
       names the session after this subject. A wrapper that gives its session to
       another subject names it last, so the score names the wrapper. If the body
       raises, the score gets nothing of the session, apart from its intervention
       counts: a change it left active still fails the release check.

    The body does neither duty itself. ``super().participate(session)`` inside a
    body reaches the parent's body directly: the outer call already holds the
    claim and has declared the inputs. A call through ``self.participate`` with
    the same session on the same thread, after the declaration, also runs the
    body directly. A subject that gives its session to another subject: that
    subject claims itself and narrows the inputs further to its own declaration;
    it never reopens an input.

    Limit: only a read through the instance gets the duties. These bypass them
    on purpose and are not supported for evaluation:
    ``type(subject).participate(subject, session)``,
    ``object.__getattribute__(subject, 'participate')``, and a subclass that
    overrides ``__getattribute__`` without calling ``super().__getattribute__``.

    ``reset`` is a lifecycle hook, not an evaluation method. Stateful subjects
    override it to clear memory, task setup and interventions between independent
    runs. A ``begin`` control event is narrower: it clears only the history ``h``
    (earlier trials, steps and responses, and the episode state of the subject's
    parts); interventions and task setup stay.
    """

    def __getattribute__(self, name):
        # super(): the next lookup in the method resolution order, so a mixin's own lookup runs.
        if name != 'participate':       # every other name: Python's normal lookup, unchanged
            return super().__getattribute__(name)
        try:
            value = super().__getattribute__(name)
        except AttributeError:
            # Python would now ask a fallback __getattr__, outside this method, and its answer
            # would get no duties. A TypeError stops that: participate must be found by the
            # normal lookup.
            raise TypeError(
                f'{type(self).__name__} has no participate that the normal lookup finds (an unset slot, '
                'or a property that raises AttributeError); a fallback __getattr__ is not used for '
                'participate. Define it as a method: def participate(self, session).') from None
        # Python's own lookup chose the body (class body, parent, mixin or an
        # assignment); the base adds the session duties around it.
        if not callable(value):
            raise TypeError(f'{type(self).__name__}.participate is {value!r}, which cannot be called; '
                            'define it as a method: def participate(self, session).')
        return functools.partial(_take_part, self, value)

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
    def output_inputs(self) -> Mapping[str, FrozenSet[str]]:
        """For an output channel, the input channels it can be answered from.

        The subject answers a listed output when it takes in at least one of its
        inputs. An output that is not listed can be answered from any input the
        subject takes, except ``perturbation``: a record of an intervention, it is
        answered only from the intervention inputs the subject takes in. A listed
        output must be in ``out_channels`` and its inputs in ``in_channels``;
        ``perturbation`` may list only intervention inputs (pre-flight raises
        ``ValueError`` otherwise). None is listed by default.
        """
        return {}

    @property
    def traits(self) -> Set[str]:
        """Properties of the subject a benchmark may require; none by default.

        ``'memory:trial_history'``: the subject keeps earlier trials, earlier
        steps and its responses from ``begin`` until ``end``. A subject without
        it always starts a trial with empty history. A benchmark that needs a
        trait lists it in ``required_traits`` (a ``history`` trial memory
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
