"""The form a behavioral answer takes.

A ``TaskContext`` says what the subject is asked to do; its ``response`` says
what form the answer takes. A response knows how to phrase itself after the
instruction (``directive``), how to read a reply (``parse``), and which output
channels it produces. A benchmark may subclass a response to change either;
prompt wording and parsing are the benchmark's science, not the model's.

Reported answers come from the model's reply. Implied measures come from the
model's internals by a stated link: a readout's or a language model's
probabilities over the labels (``behavior:probabilities``). Every result says
which it is.
"""
import math
import re
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class Scale:
    """A number on a scale, such as a 1 to 7 rating. ``step=None`` allows any value."""
    low: float
    high: float
    step: Optional[float] = None

    channels = ('behavior:rating',)

    def directive(self) -> str:
        whole = self.step is not None and float(self.step).is_integer()
        bounds = (f'{int(self.low)} to {int(self.high)}' if whole
                  else f'{self.low} to {self.high}')
        return f'Answer with a number from {bounds}. Reply with only that number.'

    def parse(self, reply: str) -> Tuple[Optional[float], Optional[str]]:
        found = re.search(r'-?\d+(?:\.\d+)?', reply or '')
        if found is None:
            return None, 'no number in the reply'
        value = float(found.group())
        if not self.low <= value <= self.high:
            return None, f'{value} is outside {self.low} to {self.high}'
        if self.step is not None and abs((value - self.low) / self.step
                                         - round((value - self.low) / self.step)) > 1e-9:
            return None, f'{value} is not on the scale step {self.step}'
        return value, None


@dataclass(frozen=True)
class Choice:
    """One label from ``labels``.

    ``probabilities=True`` also returns the implied probability of each label
    (from a readout or a language model's likelihoods). ``confidence=Scale(...)``
    also asks the subject to report how sure it is, in the same reply.
    ``reaction_time`` also returns an implied reaction time from those
    probabilities, scored by correlation with human RTs as Futrell2018 scores
    surprisal: ``'evidence_margin'`` (1 minus the gap between the two most
    probable labels; close calls are slow) or ``'surprisal'`` (-log2 of the
    chosen label's probability).

    ``labels`` is a sequence of at least two option strings, or one string:
    the name of a stimulus column whose cell holds the options of that row,
    when the options differ from row to row. Every row must then give the same
    number of distinct options. The ``choice`` axis of the answers is then the
    position of each option in the row's cell, and ``Prompt.response`` holds
    that row's options as ``labels``.

    ``raw_scores=True`` returns on ``behavior:probabilities`` the score of each
    option exactly as the model's ``likelihood`` part gave it (the summed log
    probability of the option's tokens), not normalized; the measure attributes
    say ``'scale': 'log'``. Only a ``likelihood`` part gives raw scores.

    ``raw_text=True`` shows the stimulus text exactly as given: no directive is
    added, and an instruction is refused. A part that scores options scores
    ``f"{text} {option}"`` as plain text, with no chat template and no
    beginning-of-sequence token, as EWoK's ``score`` does. A ``reply`` part
    still answers through its own chat template, because a chat model cannot
    answer without one; it gets no added sentence.
    """
    labels: Union[Tuple[str, ...], str, None] = None   # the options, or the column holding each row's
    probabilities: bool = False
    confidence: Optional[Scale] = None
    reaction_time: Optional[str] = None   # 'evidence_margin' or 'surprisal': an implied RT
    raw_scores: bool = False
    raw_text: bool = False

    REACTION_TIMES = ('evidence_margin', 'surprisal')

    def __post_init__(self):
        if isinstance(self.labels, str):
            if not self.labels:
                raise ValueError('A column name in labels must not be empty')
        elif self.labels is None or len(self.labels) < 2:
            raise ValueError('Choice needs at least two labels')
        if self.reaction_time not in (None, *self.REACTION_TIMES):
            raise ValueError(f'reaction_time is one of {self.REACTION_TIMES}, not {self.reaction_time!r}')
        for name in ('probabilities', 'raw_scores', 'raw_text'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} is True or False, not {getattr(self, name)!r}')
        if self.raw_scores and self.reaction_time is not None:
            raise ValueError('reaction_time reads normalized probabilities; it cannot use raw_scores')
        if self.confidence is not None and (self.raw_scores or self.raw_text):
            raise ValueError('A reported confidence needs a reply with a directive; '
                             'it cannot be used with raw_scores or raw_text')
        if not isinstance(self.labels, str):
            object.__setattr__(self, 'labels', tuple(self.labels))

    @property
    def channels(self) -> Tuple[str, ...]:
        return (('behavior:choice',)
                + (('behavior:probabilities',) if self.probabilities or self.raw_scores else ())
                + (('behavior:confidence',) if self.confidence is not None else ())
                + (('behavior:rt',) if self.reaction_time is not None else ()))

    def directive(self) -> str:
        if self.raw_text:
            return ''
        if isinstance(self.labels, str):
            raise ValueError("The directive needs the row's options; trial_prompt gives them")
        text = f'Answer with exactly one of: {", ".join(self.labels)}.'
        if self.confidence is not None:
            text += (f' Then, on a new line, rate your confidence from '
                     f'{self.confidence.low:g} to {self.confidence.high:g}.')
        return text

    def parse(self, reply: str) -> Tuple[Optional[str], Optional[str]]:
        first = (reply or '').strip().splitlines()[0].strip().lower() if (reply or '').strip() else ''
        for label in self.labels:
            if first == label.lower():
                return label, None
        found = [label for label in self.labels
                 if re.search(rf'\b{re.escape(label.lower())}\b', first)]
        if len(found) == 1:
            return found[0], None
        return None, ('no label in the reply' if not found
                      else f'several labels in the reply: {found}')

    def parse_confidence(self, reply: str) -> Tuple[Optional[float], Optional[str]]:
        lines = (reply or '').strip().splitlines()
        if len(lines) < 2:
            return None, 'no confidence line in the reply'
        return self.confidence.parse(' '.join(lines[1:]))


@dataclass(frozen=True)
class Text:
    """A written answer, kept as written. Brain-Score never scores text itself:
    the benchmark brings its own parser, judge or comparison.

    The generation settings are part of the task, so the benchmark sets them
    and a replying model obeys them: ``max_tokens`` (the longest reply, in
    tokens), ``temperature`` (0 or ``None`` means greedy decoding), ``top_p``
    (needs a temperature above 0) and ``seed``. ``None`` leaves the model's
    own default. A ``reply`` part reads them from ``prompt.response``.
    """
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None

    channels = ('behavior:text',)

    def __post_init__(self):
        if self.max_tokens is not None and (type(self.max_tokens) is not int or self.max_tokens < 1):
            raise ValueError(f'max_tokens is a positive integer, not {self.max_tokens!r}')
        if self.temperature is not None and (
                isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float))
                or not math.isfinite(self.temperature) or self.temperature < 0):
            raise ValueError(f'temperature is a finite number of 0 or more, not {self.temperature!r}')
        if self.top_p is not None:
            if (isinstance(self.top_p, bool) or not isinstance(self.top_p, (int, float))
                    or not 0 < self.top_p <= 1):
                raise ValueError(f'top_p is a number above 0 and at most 1, not {self.top_p!r}')
            if not self.temperature:
                raise ValueError('top_p applies only to sampling; give a temperature above 0')
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError(f'seed is an integer, not {self.seed!r}')

    def directive(self) -> str:
        return ''

    def parse(self, reply: str) -> Tuple[Optional[str], Optional[str]]:
        return (reply, None) if isinstance(reply, str) else (None, 'the reply is not text')


# The stimulus columns that carry each kind of trial part.
PART_COLUMNS = {
    'text': ('sentence', 'text'),
    'image': ('image_file_name', 'image_path', 'filename'),
    'video': ('video_path', 'video_file_name'),
    'audio': ('audio_path', 'audio_file_name', 'audio_file'),
}


@dataclass(frozen=True)
class Prompt:
    """What one trial shows a replying model, in order.

    ``parts`` are ``(kind, content)`` pairs: ``('text', str)``, or ``('image' |
    'video' | 'audio', path)``. The order is the trial's: the instruction, the
    stimulus parts in ``TaskContext.present`` order, then the response's
    directive. A ``reply`` or ``likelihood`` part shows them in this order.
    ``response`` is the task's response; a ``Text`` response carries the
    generation settings the reply obeys. For a ``Choice`` whose ``labels``
    names a column it holds this row's options as ``labels``.
    """
    parts: Tuple[Tuple[str, Any], ...]
    stimulus_id: Optional[str] = None
    names: Tuple[str, ...] = ()   # one per part: 'instruction', a column name, or 'directive'
    history: Tuple[Tuple['Prompt', str], ...] = ()   # earlier trials of the session: (prompt, reply)
    response: Any = None   # the task's Choice, Scale or Text

    def text(self) -> str:
        """The text parts joined, for a model that reads text only."""
        return '\n\n'.join(content for kind, content in self.parts if kind == 'text')


def trial_prompt(task, stimulus_row) -> Prompt:
    """Build one trial's Prompt from the task and a stimulus row (fails before any compute)."""
    present = {}
    for kind, columns in PART_COLUMNS.items():
        for column in columns:
            value = _cell(stimulus_row, column)
            if value is not None:
                present[kind] = value
                break
    order = tuple(task.present) if task.present is not None else tuple(present)
    if task.present is None and len(present) > 1:
        raise ValueError(f'Each stimulus has several parts {sorted(present)}; say the order people '
                         'saw them in with TaskContext(present=(...)), e.g. present=("text", "video").')
    shown = []          # (kind, value, name)
    for entry in order:
        if ':' in entry:
            # A named part: 'image:word_1' shows the image in column word_1.
            kind, column = entry.split(':', 1)
            value = _cell(stimulus_row, column)
            if value is None:
                raise ValueError(f'TaskContext.present names column {column!r}, which this stimulus lacks.')
            shown.append((kind, value, column))
        elif entry in present:
            shown.append((entry, present[entry], entry))
        else:
            raise ValueError(f'TaskContext.present names {entry!r}, but the stimulus has only {sorted(present)}.')
    parts, names = [], []
    if (task.instruction or '').strip():
        parts.append(('text', task.instruction.strip())); names.append('instruction')
    for kind, value, name in shown:
        parts.append((kind, value)); names.append(name)
    response = task.response
    if isinstance(response, Choice):
        response = _row_choice(response, stimulus_row)
    directive = response.directive() if response is not None else ''
    if directive:
        parts.append(('text', directive)); names.append('directive')
    return Prompt(tuple(parts), stimulus_id=_cell(stimulus_row, 'stimulus_id'), names=tuple(names),
                  response=response)


def _row_choice(response: 'Choice', stimulus_row) -> 'Choice':
    """``response`` with the row's own options as ``labels`` (when ``labels`` names a column)."""
    if not isinstance(response.labels, str):
        return response
    column = response.labels
    options = _cell(stimulus_row, column)
    if (options is None or isinstance(options, str)
            or not all(isinstance(option, str) for option in options)):
        raise ValueError(f'Column {column!r} must hold a sequence of option '
                         f'strings in every row; got {options!r}')
    options = tuple(options)
    if len(set(options)) != len(options):
        raise ValueError(f'The options in column {column!r} must be distinct; '
                         f'got {options!r}')
    return replace(response, labels=options)


def _cell(row, column):
    try:
        value = row[column]
    except (KeyError, TypeError, IndexError):
        return None
    if value is None or (isinstance(value, float) and value != value):
        return None
    return value


def response_channels(response) -> Tuple[str, ...]:
    return () if response is None else tuple(response.channels)


def prompt_for(instruction: Optional[str], response) -> str:
    """The text a replying model receives: the instruction, then the directive."""
    parts = [part for part in ((instruction or '').strip(), response.directive()) if part]
    return '\n\n'.join(parts)


def labels_of(response) -> Optional[Sequence[str]]:
    """The task's labels; ``None`` when there are none or each row gives its own."""
    if isinstance(response, Choice) and not isinstance(response.labels, str):
        return response.labels
    return None
