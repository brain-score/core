"""The form a behavioral answer takes.

A ``TaskContext`` says what the subject is asked to do; its ``response`` says
what form the answer takes and which output channels it produces. ``Choice``,
``Scale``, ``Text`` and ``Segments`` are the core response classes; a task takes these classes
only, not a subclass.

``directive`` and ``parse`` are the model side's tools; a benchmark gives the
specification only (the options, the scale, the instruction). A model part that
generates text phrases the answer line with ``directive`` and reads its reply
with ``parse``; each format defines both once, here.

An answer comes from the model's reply, a likelihood part, a readout or a
stated link. Every answer says which in its response record
(``brainscore_core.response_formats.response_record``).
"""
import math
import re
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence, Tuple, Union


def _on_step(value: float, low: float, step: float) -> bool:
    """Whether ``value`` is on the step of a scale.

    The grid of a scale is decimal: the points Decimal(repr(low)) + n * Decimal(repr(step)).
    A value is on the step when its decimal index is within 1e-9 of an integer n, or when
    it is the float of the grid point n.
    """
    from decimal import Decimal, localcontext
    with localcontext() as context:
        context.prec = 50
        low_d, step_d = Decimal(repr(float(low))), Decimal(repr(float(step)))
        index = (Decimal(repr(float(value))) - low_d) / step_d
        n = index.to_integral_value()
        return abs(index - n) <= Decimal('1e-9') or float(low_d + n * step_d) == float(value)


_NUMBER = re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?')


def _numbers(reply: str):
    """Return the exact values of the numbers in ``reply``, or a reason it is invalid.

    A run of word characters and number punctuation that contains a digit is one
    token. A token is one signed decimal number (a leading point and an exponent are
    allowed; a Unicode minus is a minus); a token with letters is a word, not a number
    (``m2``). Any other token (``1,000``, ``3,5``, ``3.5.2``) is a notation this parser
    does not support: it is refused, never read as another number. A hyphen after a
    digit separates two numbers (``3-4``).
    """
    from decimal import Decimal, InvalidOperation
    numbers = set()
    for token in re.findall(r'[\w.,+\-\u2212]+', reply):
        token = token.rstrip('.,').replace('\u2212', '-')
        if not any(c.isdigit() for c in token):
            continue
        # A hyphen after a digit joins two numbers, as in a range: ``3-4``.
        parts = re.split(r'(?<=\d)-', token)
        try:
            if all(_NUMBER.fullmatch(part) for part in parts):
                numbers.update(Decimal(part) for part in parts)
                continue
        except InvalidOperation:    # an exponent that no decimal can hold
            pass
        # A plain word with a digit (``m2``, ``3rd``) is a word. Any other token is refused.
        if not token.replace('_', 'a').isalnum():
            return set(), (f'unsupported number notation {token!r}; reply with plain '
                           'numbers such as 3, -2.5 or 1e3')
        if _NUMBER.fullmatch(token):
            return set(), f'the number {token!r} is too large or too small to read'
    return numbers, None


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
        # One number, also repeated, is the answer; several distinct numbers are invalid.
        # Numbers are compared by their exact decimal value, before any float conversion.
        numbers, reason = _numbers(reply or '')
        if reason:
            return None, reason
        if not numbers:
            return None, 'no number in the reply'
        if len(numbers) > 1:
            return None, f'several numbers in the reply: {sorted(map(str, numbers))}'
        from decimal import Decimal
        exact, = numbers
        # The bounds are checked on the exact value: a float can round a value onto a bound.
        if not Decimal(str(self.low)) <= exact <= Decimal(str(self.high)):
            return None, f'{exact} is outside {self.low} to {self.high}'
        value = float(exact)
        if self.step is not None and not _on_step(value, self.low, self.step):
            return None, f'{value} is not on the scale step {self.step}'
        return value, None


@dataclass(frozen=True)
class Choice:
    """One label from ``labels``.

    A choice task can be answered as the choice (``behavior:choice``), as
    the implied probability of each label (``behavior:probabilities``, from a
    readout or a language model's likelihoods) and as the log probability of
    each option (``behavior:log_probabilities``). ``channels`` names every format
    the task can be answered in; the outputs that a session requests say which
    are answered. ``reaction_time`` also returns an implied reaction time from
    those probabilities, scored by correlation with human RTs as Futrell2018
    scores surprisal: ``'evidence_margin'`` (1 minus the gap between the two most
    probable labels; close calls are slow) or ``'surprisal'`` (-log2 of the
    chosen label's probability).

    ``labels`` is a sequence of at least two option strings, or one string:
    the name of a stimulus column whose cell holds the options of that row,
    when the options differ from row to row. Every row must then give the same
    number of distinct options. The ``choice`` axis of the answers is then the
    position of each option in the row's cell, and ``Prompt.response`` holds
    that row's options as ``labels``.

    The log probability of each option, exactly as the model's ``likelihood``
    part gives it (``behavior:log_probabilities``), is requested by a session;
    with ``verbatim=True`` the part scores the plain text, as EWoK does.

    ``verbatim=True`` shows the trial text exactly as given: no directive is
    added, and an instruction is refused. A part that scores options scores
    ``f"{text} {option}"`` as plain text, with no chat template and no
    beginning-of-sequence token, as EWoK's ``score`` does. A ``reply`` part
    keeps its own chat template, because a chat model cannot answer without one;
    it gets no added sentence, and its answer's record notes its own template,
    if it has one.
    """
    labels: Union[Tuple[str, ...], str, None] = None   # the options, or the column holding each row's
    reaction_time: Optional[str] = None   # 'evidence_margin' or 'surprisal': an implied RT
    verbatim: bool = False

    REACTION_TIMES = ('evidence_margin', 'surprisal')

    def __post_init__(self):
        if isinstance(self.labels, str):
            if not self.labels:
                raise ValueError('A column name in labels must not be empty')
        elif self.labels is None or len(self.labels) < 2:
            raise ValueError('Choice needs at least two labels')
        if self.reaction_time not in (None, *self.REACTION_TIMES):
            raise ValueError(f'reaction_time is one of {self.REACTION_TIMES}, not {self.reaction_time!r}')
        for name in ('verbatim',):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} is True or False, not {getattr(self, name)!r}')
        if not isinstance(self.labels, str):
            object.__setattr__(self, 'labels', tuple(self.labels))

    @property
    def channels(self) -> Tuple[str, ...]:
        return (('behavior:choice', 'behavior:probabilities', 'behavior:log_probabilities')
                + (('behavior:rt',) if self.reaction_time is not None else ()))

    def directive(self) -> str:
        if self.verbatim:
            return ''
        if isinstance(self.labels, str):
            raise ValueError("The directive needs the row's options; trial_prompt gives them")
        return f'Answer with exactly one of: {", ".join(self.labels)}.'

    def parse(self, reply: str) -> Tuple[Optional[str], Optional[str]]:
        if isinstance(self.labels, str):
            raise ValueError(f"labels is still the column name {self.labels!r}; parse with the "
                             "row's options (trial_prompt resolves them per row).")
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


@dataclass(frozen=True)
class Segments:
    """The log probability of each consecutive part of one text, given the earlier parts.

    ``segments`` is the name of a stimulus column whose cell holds the ordered
    parts of that row (``trial_prompt`` resolves it to the row's parts, a tuple of
    str). The row's text is the parts joined by ``brainscore_core.text.prepare_context``;
    a row whose text differs from that join is refused. The answer is
    ``behavior:log_probabilities`` in the segment layout: position i holds the
    summed log probability of the new tokens of part i given parts 1 to i-1 of the
    same row (Response Format Standard 2.3). Parts may repeat; an empty part (no text
    other than white space) is refused: an empty part cannot be scored.
    """
    segments: Union[str, Tuple[str, ...]]

    channels = ('behavior:log_probabilities',)

    def __post_init__(self):
        if isinstance(self.segments, str):
            if not self.segments:
                raise ValueError('segments names a column; it must not be empty')
            return
        parts = tuple(self.segments)
        if not parts or not all(isinstance(part, str) for part in parts):
            raise ValueError(f'segments is a column name or a nonempty sequence of str, '
                             f'not {self.segments!r}')
        empty = [i + 1 for i, part in enumerate(parts) if not part.strip()]
        if empty:
            raise ValueError(f'an empty part cannot be scored: positions {empty} have no text')
        object.__setattr__(self, 'segments', parts)

    def directive(self) -> str:
        return ''


# The channel that carries each kind of trial part.
PART_CHANNELS = {'text': 'text', 'image': 'vision:image', 'video': 'vision:video', 'audio': 'audio'}


def part_columns(kind: str) -> Tuple[str, ...]:
    """The stimulus columns that carry a part of ``kind``, in the order a prompt looks.

    They are the columns that feed the part's channel in the catalog's column
    table (``io_catalog.stimulus_columns``).
    """
    from .io_catalog import stimulus_columns
    channel = PART_CHANNELS[kind]
    return tuple(column for column, fed in stimulus_columns().items() if fed == channel)


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
    for kind in PART_CHANNELS:
        for column in part_columns(kind):
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
    elif isinstance(response, Segments):
        response = _row_segments(response, stimulus_row, [content for kind, content in parts
                                                          if kind == 'text'])
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


def _row_segments(response: 'Segments', stimulus_row, texts) -> 'Segments':
    """``response`` with the row's own parts; the row's text must be their join."""
    from .text import prepare_context
    resolved = response
    if isinstance(response.segments, str):
        parts = _cell(stimulus_row, response.segments)
        if parts is None or isinstance(parts, str):
            raise ValueError(f'Column {response.segments!r} must hold a sequence of str in every '
                             f'row; got {parts!r}')
        resolved = Segments(tuple(parts))
    if texts != [prepare_context(resolved.segments)]:
        raise ValueError(f'The text of a Segments row is the join of its parts '
                         f'({prepare_context(resolved.segments)!r}); got {texts!r}')
    return resolved


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


def labels_of(response) -> Optional[Sequence[str]]:
    """The task's labels; ``None`` when there are none or each row gives its own."""
    if isinstance(response, Choice) and not isinstance(response.labels, str):
        return response.labels
    return None
