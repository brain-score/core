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
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple


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
    """
    labels: Tuple[str, ...]
    probabilities: bool = False
    confidence: Optional[Scale] = None

    def __post_init__(self):
        if isinstance(self.labels, str) or len(self.labels) < 2:
            raise ValueError('Choice needs at least two labels')
        object.__setattr__(self, 'labels', tuple(self.labels))

    @property
    def channels(self) -> Tuple[str, ...]:
        return (('behavior:choice',)
                + (('behavior:probabilities',) if self.probabilities else ())
                + (('behavior:confidence',) if self.confidence is not None else ()))

    def directive(self) -> str:
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
    the benchmark brings its own parser, judge or comparison."""
    channels = ('behavior:text',)

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
    """
    parts: Tuple[Tuple[str, Any], ...]
    stimulus_id: Optional[str] = None
    names: Tuple[str, ...] = ()   # one per part: 'instruction', a column name, or 'directive'
    history: Tuple[Tuple['Prompt', str], ...] = ()   # earlier trials of the session: (prompt, reply)

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
    directive = task.response.directive() if task.response is not None else ''
    if directive:
        parts.append(('text', directive)); names.append('directive')
    return Prompt(tuple(parts), stimulus_id=_cell(stimulus_row, 'stimulus_id'), names=tuple(names))


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
    return response.labels if isinstance(response, Choice) else None
