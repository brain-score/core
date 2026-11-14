"""Behavioral response formats in core: conversions, the response record, storage.

The benchmark specifies the response; the model side produces it; core
validates it; the benchmark scores it (Response Format Standard, section 0).

- **Conversions**: ``argmax`` (probabilities to a choice), ``softmax``
  (continuation log-probabilities to probabilities) and ``softmax_argmax``
  (both in turn). Each returns the validity of every row with its result: a
  non-finite source row, or a row that is not a valid probability row for
  ``argmax``, gives an invalid result. Stage 1 exception: a finite positive
  log value passes through ``softmax`` as valid.
- **The response record** (``response_record``): the dict an answer carries
  in ``attrs['response']``. Scores copy the records of their answers.
- **Storage**: the ``behavior`` catalog entry's validator checks a response
  on an addressed ``behavior:<format>`` channel against the session's task and
  the trials it names, and the session stores the checked answer in the schema
  of its format. The bare ``behavior`` channel is not checked or stored here
  (legacy handling).

What an addressed response can be: a cell for one trial (``behavior:text``
and ``behavior:choice``: a ``str``; ``behavior:probabilities``: a 1-D array
with one value per option; ``behavior:rating``, ``behavior:confidence``,
``behavior:rt``: a number), or an assembly of rows over ``presentation``
with a ``stimulus_id`` coordinate and a response record. A cell is always
native: core gives it a record with ``method='external'`` (for ``behavior:rt``
the units come from ``event.meta['units']``). A derived answer is an assembly
(one row is fine) with its record and its source evidence in ``raw_answer``.
A malformed response raises ``ValueError`` and is never stored.

The stored answer keeps its rows in the order they arrived. Each row has
``raw_answer``, ``response_state`` (``valid``, ``invalid``, ``missing``) and
``response_reason``; the attrs hold the record, except for an answer whose
trials were all missing (nothing answered, so no record). An answer is
expected for each row of a session's table (by occurrence: the k-th answer
row with stimulus id X answers the k-th table row with id X) and, in a
session without a table, for each ``respond`` signal of the source. A
requested trial with no answer is stored as a missing row once the session is
complete; a session with no table and no ``respond`` signal requests no trial,
so nothing is missing there.
"""
import json
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_TOLERANCE = 1e-4          # absolute, on the sum of a probability row
_STEP_TOLERANCE = 1e-9     # on (value - low) / step, for a rating scale with a step
_CONVERSIONS = ('argmax', 'softmax', 'softmax_argmax')
_CATEGORIES = ('behavioral', 'model-derived')
_ORIGINS = ('native', 'derived')
_PARSERS = ('model', 'library')
_UNITS = ('nats', 'bits')
_RT_UNITS = ('ms', 'monotone')
_FIELDS = frozenset(('format', 'category', 'origin', 'method', 'conversion', 'source_format',
                     'parameters', 'parsed_by'))
_FORMATS = ('choice', 'probabilities', 'rating', 'confidence', 'rt', 'text')
# The format each conversion reads.
_SOURCES = {'argmax': 'behavior:probabilities', 'softmax': 'behavior:log_probabilities',
            'softmax_argmax': 'behavior:log_probabilities'}


# --------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------

def _valid_probability_rows(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values).all(axis=1)
    with np.errstate(invalid='ignore'):
        positive = (values >= 0).all(axis=1)
        summed = np.abs(values.sum(axis=1) - 1.0) <= _TOLERANCE
    return finite & positive & summed


def argmax(probabilities) -> Tuple[np.ndarray, np.ndarray]:
    """The most probable option of each row: ``(indices, valid)``.

    Ties go to the first maximum in the given column order (numpy ``argmax``);
    the column order is the caller's, and its record says which order it is.
    Stage 1 exception: a readout passes its own column order (``option_order``
    ``'readout'``), not the trial's.
    A row that is not a valid probability row (a non-finite or negative value,
    or a sum that differs from 1 by more than 1e-4) has no choice: its index
    is -1 and ``valid`` is False. The result is the most probable option under
    the model's distribution, not a sampled choice.
    """
    values = np.asarray(probabilities)
    if values.ndim != 2:
        raise ValueError(f'argmax takes a (trials, options) array; got shape {values.shape}')
    valid = _valid_probability_rows(values)
    indices = np.where(valid, values.argmax(axis=1), -1)
    return indices, valid


def softmax(log_probabilities, units: str = 'nats') -> Tuple[np.ndarray, np.ndarray]:
    """Each row of log-probabilities as probabilities over the row's options: ``(p, valid)``.

    Temperature 1, in the given column order. ``units`` is the unit of the
    source, ``'nats'`` (natural log) or ``'bits'`` (log base 2; multiplied by
    ln 2 first). The row's offset is lost: two rows that differ by a constant
    give the same probabilities, a distribution over the listed continuations
    only. A source row with a non-finite value (NaN, +inf or -inf) is invalid,
    and so is its result; its values are computed as for any row. Stage 1
    exception: a finite positive log value passes as valid.
    """
    if units not in _UNITS:
        raise ValueError(f'units is one of {_UNITS}, not {units!r}')
    logp = np.asarray(log_probabilities)
    if logp.ndim != 2:
        raise ValueError(f'softmax takes a (trials, options) array; got shape {logp.shape}')
    valid = np.isfinite(logp).all(axis=1)
    if units == 'bits':
        logp = logp * math.log(2)
    logp = logp - logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True), valid


def softmax_argmax(log_probabilities, units: str = 'nats') -> Tuple[np.ndarray, np.ndarray]:
    """``argmax`` of ``softmax``: the most likely listed continuation, ``(indices, valid)``.

    A row is valid only when its source row is valid and its softmax row is.
    """
    probabilities, source_valid = softmax(log_probabilities, units)
    indices, valid = argmax(probabilities)
    valid = valid & source_valid
    return np.where(valid, indices, -1), valid


_DEFAULT_PARAMETERS = {
    'argmax': {'tie': 'first', 'option_order': 'trial'},
    'softmax': {'temperature': 1, 'option_order': 'trial', 'units': 'nats'},
    'softmax_argmax': {'temperature': 1, 'tie': 'first', 'option_order': 'trial', 'units': 'nats'},
}


# --------------------------------------------------------------------------
# The response record
# --------------------------------------------------------------------------

def response_record(format: str, method: str, *, conversion: Optional[str] = None,
                    source_format: Optional[str] = None, parameters: Optional[Dict] = None,
                    parsed_by: Optional[str] = None,
                    category: str = 'behavioral') -> Dict[str, Any]:
    """The record of one answer: put it in the answer's ``attrs['response']``.

    ``method`` says how the model side gave the answer (``generation``,
    ``likelihood``, ``readout``, ``link``, ``external``; a subject that is not
    a ``BrainScoreModel`` names its own). An answer is ``derived`` when it has a
    conversion or is a link (``method='link'``); a derived answer names its
    ``source_format``. A conversion's ``parameters`` default to its stated
    ones. ``parsed_by`` is ``'model'`` when the model side parsed generated
    text, ``'library'`` when an external library parses it.
    """
    if parameters is None:
        parameters = dict(_DEFAULT_PARAMETERS.get(conversion, {}))
    derived = conversion is not None or method == 'link'
    record = {'format': format, 'category': category, 'origin': 'derived' if derived else 'native',
              'method': method, 'conversion': conversion, 'source_format': source_format,
              'parameters': dict(parameters), 'parsed_by': parsed_by}
    problems = _record_problems(record, format)
    if problems:
        raise ValueError('; '.join(problems))
    return record


def _plain(value) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_plain(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _plain(item) for key, item in value.items())
    return False


def _record_problems(record, channel: str) -> List[str]:
    """Why ``record`` is not a well-formed response record for ``channel``; empty when it is."""
    if not isinstance(record, dict):
        return [f'the response record is {type(record).__name__}, not a dict']
    if set(record) != _FIELDS:
        return [f'the response record has fields {sorted(record)}, not {sorted(_FIELDS)}']
    problems = []
    if record['format'] != channel:
        problems.append(f"the response record names format {record['format']!r}, "
                        f'but the answer is on {channel!r}')
    if record['category'] not in _CATEGORIES:
        problems.append(f"the record's category {record['category']!r} is not one of {_CATEGORIES}")
    if not isinstance(record['method'], str) or not record['method']:
        problems.append(f"the record's method {record['method']!r} is not a nonempty string")
    if record['conversion'] not in (None, *_CONVERSIONS):
        problems.append(f"the record's conversion {record['conversion']!r} is not one of {_CONVERSIONS}")
    if record['parsed_by'] not in (None, *_PARSERS):
        problems.append(f"the record's parsed_by {record['parsed_by']!r} is not one of {_PARSERS}")
    derived = record['conversion'] is not None or record['method'] == 'link'
    if record['origin'] != ('derived' if derived else 'native'):
        problems.append(f"the record's origin {record['origin']!r} does not match its conversion "
                        f"{record['conversion']!r} and method {record['method']!r}")
    if derived != bool(record['source_format']):
        problems.append('a derived answer, and only a derived answer, names its source_format')
    elif derived:
        expected = _SOURCES.get(record['conversion'])
        if expected is not None and record['source_format'] != expected:
            problems.append(f"the {record['conversion']} conversion reads {expected!r}, not "
                            f"{record['source_format']!r}")
        elif expected is None and _format(record['source_format']) is None:
            problems.append(f"a link names the behavior format it was computed from, not "
                            f"{record['source_format']!r}")
    if not isinstance(record['parameters'], dict) or not _plain(record['parameters']):
        problems.append("the record's parameters are not a dict of plain values")
    else:
        try:
            json.dumps(record['parameters'], allow_nan=False)
        except ValueError:
            problems.append("the record's parameters are not serializable")
    return problems


# --------------------------------------------------------------------------
# Storage: the checks of a session, and the stored schema
# --------------------------------------------------------------------------

def _format(channel) -> Optional[str]:
    """The format of an addressed behavior channel, or None."""
    if isinstance(channel, str) and channel.startswith('behavior:'):
        address = channel.split(':', 1)[1]
        return address if address in _FORMATS else None
    return None


def _number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _response_of(session):
    return getattr(getattr(session, 'task', None), 'response', None)


def _table(session):
    table = getattr(session, 'stimuli', None)
    return table if table is not None and 'stimulus_id' in getattr(table, 'columns', ()) else None


class _Ledger:
    """What a session's table expects and what each channel has answered.

    Built once per session: the occurrence count and the table positions of
    each stimulus id, the per-row options, and for each channel the occurrences
    answered so far.
    ``pending`` keeps the rows the validator built for the event the session is
    accepting, so each response is checked and packaged in one pass.
    """

    def __init__(self, session):
        table = _table(session)
        self.ids = None if table is None else list(table['stimulus_id'])
        self.counts = None if table is None else Counter(self.ids)
        self.positions: Dict[Any, list] = {}
        for position, stimulus_id in enumerate(self.ids or ()):
            self.positions.setdefault(stimulus_id, []).append(position)
        self.answered: Dict[str, Counter] = {}
        self.pending = None
        self._session = session
        self._column = None

    def options(self, stimulus_id, occurrence: int = 0) -> Optional[list]:
        """The options of the ``occurrence``-th trial (from 0) with this stimulus id.

        The k-th answer for an id answers the k-th table row with that id, so
        per-row options are read from that row, not from the first one.
        """
        from .responses import Choice
        response = _response_of(self._session)
        if isinstance(response, Choice):
            if not isinstance(response.labels, str):
                return list(response.labels)
            positions = self.positions.get(stimulus_id, ())
            if occurrence >= len(positions):
                return None
            if self._column is None:
                self._column = _table(self._session)[response.labels].tolist()
            return list(self._column[positions[occurrence]])
        label_set = getattr(getattr(self._session, 'task', None), 'label_set', None)
        return list(label_set) if label_set else None

    def occurrence(self, channel, stimulus_id) -> int:
        """How many rows of ``channel`` already answer this stimulus id."""
        return self.answered.get(channel, Counter())[stimulus_id]


def _ledger(session) -> Optional[_Ledger]:
    if session is None:
        return None
    ledger = session.__dict__.get('_formats_ledger')
    if ledger is None:
        ledger = session.__dict__['_formats_ledger'] = _Ledger(session)
    return ledger


def _axis(session, options) -> Optional[list]:
    """The choice axis: the shared labels, or positions when options differ by row."""
    from .responses import Choice
    if options is None:
        return None
    if isinstance(_response_of(session), Choice) and isinstance(_response_of(session).labels, str):
        return list(range(len(options)))
    return list(options)


def _presentation_coords(payload) -> Dict[str, list]:
    names = []
    index = payload.indexes.get('presentation') if hasattr(payload, 'indexes') else None
    for name in getattr(index, 'names', None) or ():
        if name and name != 'presentation':
            names.append(name)
    for name, coord in payload.coords.items():
        if coord.dims == ('presentation',) and name != 'presentation' and name not in names:
            names.append(name)
    coords = {name: list(payload[name].values) for name in names}
    if not coords and index is not None and not getattr(index, 'nlevels', 1) > 1:
        # An assembly with one presentation coordinate keeps it as the index of
        # the dimension: that coordinate is stimulus_id.
        coords['stimulus_id'] = list(index)
    return coords


def _null(value) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _raw_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        return str([float(v) for v in value.ravel()])
    return str(value)


def _value_reasons(fmt, values, record, response) -> List[Optional[str]]:
    """Why each trial's value is not a usable answer (None when it is)."""
    reasons: List[Optional[str]] = []
    if fmt == 'choice':
        for row in values:
            if np.isnan(row).all():
                reasons.append('no option chosen')
            else:
                reasons.append(None if (row == 1).sum() == 1 else 'not exactly one option chosen')
    elif fmt == 'probabilities':
        raw_scores = (record['category'] == 'model-derived'
                      and record['parameters'].get('scale') == 'log')
        for row in values:
            row = row.astype(np.float64)
            if not np.isfinite(row).all():
                reasons.append('a value is not finite')
            elif raw_scores:
                reasons.append('a log probability is above 0' if (row > 0).any() else None)
            elif (row < 0).any():
                reasons.append('a probability is negative')
            elif abs(float(row.sum()) - 1.0) > _TOLERANCE:
                reasons.append(f'the probabilities sum to {float(row.sum())!r}, not 1')
            else:
                reasons.append(None)
    elif fmt in ('rating', 'confidence'):
        scale = getattr(response, 'confidence', None) if fmt == 'confidence' else response
        low, high = getattr(scale, 'low', None), getattr(scale, 'high', None)
        step = getattr(scale, 'step', None)
        for value in values:
            value = float(value)
            if not np.isfinite(value):
                reasons.append('no number')
            elif low is not None and not low <= value <= high:
                reasons.append(f'{value} is outside {low} to {high}')
            elif step is not None and abs((value - low) / step
                                          - round((value - low) / step)) > _STEP_TOLERANCE:
                reasons.append(f'{value} is not on the scale step {step}')
            else:
                reasons.append(None)
    elif fmt == 'rt':
        for value in values:
            value = float(value)
            reasons.append('not finite' if not np.isfinite(value)
                           else 'negative' if value < 0 else None)
    else:   # text: a str, or None for no reply (missing)
        reasons = [None] * len(values)
    return reasons


def _cell_rows(fmt, channel, payload, event, session, ledger):
    """A cell as ``(values, coords, record)`` of a one-row stored answer."""
    meta = getattr(event, 'meta', None) or {}
    stimulus_id = meta.get('stimulus_id')
    options = (ledger.options(stimulus_id, ledger.occurrence(channel, stimulus_id))
               if ledger is not None and fmt in ('choice', 'probabilities') else None)
    coords = {'stimulus_id': [stimulus_id], 'answer_valid': [True], 'parse_error': ['']}
    parameters = {}
    if fmt in ('text', 'choice'):
        if not isinstance(payload, str):
            raise ValueError(f'a {channel} cell is a str, not {type(payload).__name__}')
    if fmt == 'text':
        values = np.array([payload], dtype=object)
    elif fmt == 'choice':
        if options is None:
            raise ValueError(f'{channel}: the options of this trial are not known; send an '
                             'assembly with a choice axis or give the task a Choice')
        row = np.zeros(len(options), dtype=np.float32)
        if payload in options:
            row[options.index(payload)] = 1.0
        else:
            row[:] = np.nan
            coords['answer_valid'], coords['parse_error'] = (
                [False], [f'{payload!r} is not one of the options {options}'])
        values = row[None, :]
    elif fmt == 'probabilities':
        array = np.asarray(payload) if isinstance(payload, np.ndarray) else None
        if array is None or array.ndim != 1 or array.dtype.kind not in 'fiu':
            raise ValueError(f'a {channel} cell is a 1-D numeric array')
        if options is not None and len(array) != len(options):
            raise ValueError(f'{channel}: {len(array)} values for {len(options)} options')
        values = array.astype(np.float32)[None, :]
    else:
        if isinstance(payload, np.ndarray) and payload.ndim == 0:
            payload = payload.item()
        if not _number(payload):
            raise ValueError(f'a {channel} cell is one number, not {type(payload).__name__}')
        values = np.array([float(payload)], dtype=np.float64)
        if fmt == 'rt':
            parameters['units'] = meta.get('units')
    coords['raw_answer'] = [_raw_text(payload)]
    record = response_record(channel, 'external', parameters=parameters)
    if fmt in ('choice', 'probabilities'):
        coords['choice'] = _axis(session, options) or list(range(values.shape[1]))
    return values, coords, record


def _assembly_rows(fmt, channel, payload, event, session, ledger):
    """An assembly as ``(values, coords, record)``, after the malformed checks."""
    dims = ('presentation', 'choice') if fmt in ('choice', 'probabilities') else ('presentation',)
    if tuple(payload.dims) != dims:
        raise ValueError(f'{channel} payload dims are {tuple(payload.dims)}, not {dims}')
    record = payload.attrs.get('response')
    if record is None:
        raise ValueError(f'a {channel} assembly carries its response record in attrs["response"]')
    problems = _record_problems(record, channel)
    if problems:
        raise ValueError('; '.join(problems))
    raw_scores = (fmt == 'probabilities' and record['category'] == 'model-derived'
                  and record['parameters'].get('scale') == 'log')
    values = np.asarray(payload.values)
    dtype = {'choice': np.float32, 'probabilities': np.float64 if raw_scores else np.float32,
             'text': object}.get(fmt, np.float64)
    if values.dtype != np.dtype(dtype):
        raise ValueError(f'{channel} values are {values.dtype}, not {np.dtype(dtype)}')
    coords = _presentation_coords(payload)
    if 'stimulus_id' not in coords:
        raise ValueError(f'a {channel} assembly has a stimulus_id coordinate')
    ids = coords['stimulus_id']
    named = (getattr(event, 'meta', None) or {}).get('stimulus_id')
    if named is not None:
        named = list(named) if isinstance(named, (list, tuple)) else [named]
        if ids != named:
            raise ValueError(f'{channel}: the event answers stimuli {named}, but its rows are {ids}')
    if fmt in ('choice', 'probabilities'):
        axis = list(payload['choice'].values)
        if ledger is not None:
            seen = Counter()
            expected_axes = []
            for stimulus_id in ids:   # each row by occurrence, also inside one event
                occurrence = ledger.occurrence(channel, stimulus_id) + seen[stimulus_id]
                seen[stimulus_id] += 1
                expected_axes.append((stimulus_id, _axis(session, ledger.options(stimulus_id,
                                                                                  occurrence))))
            widths = {len(expected) for _, expected in expected_axes if expected is not None}
            if len(widths) > 1:
                raise ValueError(f'{channel}: the rows of this answer need different numbers of '
                                 f'options {sorted(widths)}, so they cannot be one assembly; '
                                 'answer such trials one row at a time (a cell or a one-row '
                                 'assembly each)')
            for stimulus_id, expected in expected_axes:
                if expected is not None and [str(a) for a in axis] != [str(e) for e in expected]:
                    raise ValueError(f'{channel}: the choice axis is {axis}, not the options '
                                     f'{expected} of stimulus {stimulus_id!r}')
    if fmt == 'choice':
        known = np.isnan(values) | (values == 0) | (values == 1)
        if not known.all():
            raise ValueError(f'{channel} values are 0, 1 or NaN (one option, or none)')
        empty = np.isnan(values).all(axis=1)
        if (np.isnan(values).any(axis=1) & ~empty).any():
            raise ValueError(f'{channel}: a row is all NaN (no choice) or has no NaN')
        supplied = coords.get('response_state')
        if supplied is not None and any(state == 'invalid' and not row_empty
                                        for state, row_empty in zip(supplied, empty)):
            raise ValueError(f'{channel}: a row supplied as invalid is all NaN')
    if fmt == 'text':
        if any(not (isinstance(v, str) or v is None) for v in values):
            raise ValueError(f'{channel} values are str (or None for no reply)')
    if fmt == 'rt' and record['parameters'].get('units') not in _RT_UNITS:
        raise ValueError(f"{channel}: the record's parameters name units {_RT_UNITS}")
    if 'raw_answer' not in coords:
        if record['origin'] == 'derived':
            raise ValueError(f'a derived {channel} assembly keeps its source evidence in '
                             'raw_answer; core does not make it from the derived values')
        coords['raw_answer'] = [_raw_text(value) for value in values]
    return values, coords, record


def _states(fmt, values, coords, record, response) -> Tuple[List[str], List[str]]:
    n = len(values)
    reasons = _value_reasons(fmt, values, record, response)
    marked = coords.get('answer_valid', [True] * n)
    errors = coords.get('parse_error', [''] * n)
    states, why = [], []
    for i in range(n):
        if _null(coords['raw_answer'][i]) or (fmt == 'text' and values[i] is None):
            states.append('missing'); why.append('no answer')
        elif not marked[i] or reasons[i] is not None:
            states.append('invalid')
            why.append(str(errors[i]) if errors[i] else reasons[i] or 'marked invalid by the model side')
        else:
            states.append('valid'); why.append('')
    return states, why


def _assembly(values, coords, record, attrs=None):
    from .supported_data_standards.brainio.assemblies import BehavioralAssembly
    axis = coords.pop('choice', None)
    built = {}
    for name, column in coords.items():
        column = list(column)
        if any(_null(value) for value in column):
            # Keep a null as a null: a list of text and None would become the text 'nan'.
            column = np.array([None if _null(v) else v for v in column], dtype=object)
        built[name] = ('presentation', column)
    dims = ['presentation']
    if axis is not None:
        built['choice'] = list(axis)
        dims.append('choice')
    attrs = dict(attrs or {})
    if record is not None:
        attrs['response'] = record
    return BehavioralAssembly(values, coords=built, dims=dims, attrs=attrs)


def _check_occurrences(channel, ids, ledger) -> None:
    """Refuse a row that the table does not expect (unknown id, or one occurrence too many)."""
    if ledger is None or ledger.counts is None:
        return
    answered = ledger.answered.get(channel, Counter())
    for stimulus_id, count in Counter(ids).items():
        if _null(stimulus_id):
            raise ValueError(f'{channel}: an answer row names no stimulus_id, so it cannot be '
                             'matched to a trial of the table')
        if stimulus_id not in ledger.counts:
            raise ValueError(f'{channel}: stimulus {stimulus_id!r} is not in this session')
        if answered[stimulus_id] + count > ledger.counts[stimulus_id]:
            raise ValueError(f'{channel}: stimulus {stimulus_id!r} is answered more often than '
                             f'the table has it ({ledger.counts[stimulus_id]})')


def _rows(channel, payload, event, session):
    """The stored ``(values, coords, record, attrs)`` of a response, or ValueError."""
    fmt = _format(channel)
    ledger = _ledger(session)
    if hasattr(payload, 'dims') and hasattr(payload, 'attrs'):
        values, coords, record = _assembly_rows(fmt, channel, payload, event, session, ledger)
        if fmt in ('choice', 'probabilities'):
            coords['choice'] = list(payload['choice'].values)
        attrs = {key: value for key, value in payload.attrs.items() if key != 'response'}
    else:
        values, coords, record = _cell_rows(fmt, channel, payload, event, session, ledger)
        attrs = {}
    if fmt == 'rt' and record['parameters'].get('units') not in _RT_UNITS:
        raise ValueError(f"{channel}: give the units of a reaction time ({_RT_UNITS}): the "
                         "record's parameters['units'], or meta['units'] for one number")
    _check_occurrences(channel, coords['stimulus_id'], ledger)
    states, why = _states(fmt, values, coords, record, _response_of(session))
    supplied = coords.pop('response_state', None)
    if supplied is not None and list(supplied) != states:
        raise ValueError(f'{channel}: the supplied states {list(supplied)} contradict the '
                         f'values, which give {states}')
    coords.pop('response_reason', None)
    coords['response_state'], coords['response_reason'] = states, why
    return values, coords, record, attrs


def _validate(payload, event=None, session=None) -> List[str]:
    """The ``behavior`` catalog validator: why an addressed response is malformed.

    The rows it builds are kept for the session, which stores them without a
    second pass (``_store``).
    """
    channel = getattr(event, 'channel', None)
    if _format(channel) is None:
        if channel == 'behavior' and _number(payload):
            return ['a number is not a payload of the bare behavior channel']
        return []
    try:
        rows = _rows(channel, payload, event, session)
    except ValueError as error:
        return [str(error)]
    ledger = _ledger(session)
    if ledger is not None:
        ledger.pending = (event, payload, rows)
    return []


def _store(channel, payload, event, session):
    """The stored answer of a checked response; the table's expected occurrences are counted."""
    ledger = _ledger(session)
    pending = ledger.pending if ledger is not None else None
    if pending is not None and pending[0] is event and pending[1] is payload:
        rows = pending[2]
    else:
        rows = _rows(channel, payload, event, session)
    if ledger is not None:
        ledger.pending = None
        ledger.answered.setdefault(channel, Counter()).update(rows[1]['stimulus_id'])
    values, coords, record, attrs = rows
    return _assembly(values, coords, record, attrs)


def _missing(channel, session, stimulus_ids: list, reason: str, template=None,
             occurrences=None):
    """A stored answer for trials with no answer: NaN (or None) values, state ``missing``.

    It has no record: nothing was answered. ``template`` (an answer of the same
    channel) gives the choice axis and the dtype; ``occurrences`` give the
    table occurrence of each id, for its options.
    """
    fmt = _format(channel)
    n = len(stimulus_ids)
    occurrences = occurrences or [0] * n
    coords = {'stimulus_id': list(stimulus_ids), 'raw_answer': [None] * n,
              'answer_valid': [False] * n, 'parse_error': [''] * n,
              'response_state': ['missing'] * n, 'response_reason': [reason] * n}
    if fmt in ('choice', 'probabilities'):
        if template is not None:
            axis = list(template['choice'].values)
        else:
            ledger = _ledger(session)
            options = (ledger.options(stimulus_ids[0], occurrences[0])
                       if ledger is not None and n else None)
            axis = _axis(session, options)
        if axis is None:
            raise ValueError(f'{channel}: trials {stimulus_ids[:5]} have no answer and their '
                             'options are not known, so they cannot be stored as missing')
        dtype = template.values.dtype if template is not None else np.float32
        values = np.full((n, len(axis)), np.nan, dtype=dtype)
        coords['choice'] = axis
    elif fmt == 'text':
        values = np.array([None] * n, dtype=object)
    else:
        values = np.full(n, np.nan, dtype=np.float64)
    return values, coords


def _table_missing(channel, session, template=None):
    """The stored missing rows of the table trials of ``channel`` with no answer, or None.

    The k-th answer row with stimulus id X answers the k-th table row with id X,
    so the missing rows are the later occurrences, in table order. The session
    calls it only once a driver has read the source to its end.
    """
    ledger = _ledger(session)
    if ledger is None or ledger.ids is None:
        return None
    answered = ledger.answered.get(channel, Counter())
    seen = Counter()
    positions, occurrences = [], []
    for position, stimulus_id in enumerate(ledger.ids):
        if seen[stimulus_id] >= answered[stimulus_id]:   # a later occurrence: no answer
            positions.append(position)
            occurrences.append(seen[stimulus_id])
        seen[stimulus_id] += 1
    if not positions:
        return None
    ids = [ledger.ids[position] for position in positions]
    if _format(channel) in ('choice', 'probabilities'):
        widths = {len(ledger.options(i, k) or ()) for i, k in zip(ids, occurrences)}
        if template is not None:
            widths.add(template.sizes['choice'])
        if len(widths) > 1:
            raise ValueError(f'{channel}: the trials of this table need different numbers of '
                             f'options {sorted(widths)}, so their answers cannot be one '
                             'assembly; read each answer from session.emitted')
    values, coords = _missing(channel, session, ids, 'no answer event', template, occurrences)
    table = _table(session)
    names = _presentation_coords(template) if template is not None else {}
    for name in names:
        if name not in coords and name in table.columns:
            coords[name] = table[name].iloc[positions].tolist()
    ledger.answered.setdefault(channel, Counter()).update(ids)   # stored once
    return _assembly(values, coords, None)


def _signal_missing(channel, session, count: int):
    """``count`` stored missing rows for respond signals that got no answer.

    Recorded limit: the rows are placed at the end of the episode and carry no
    identity of the single signal they stand for.
    """
    values, coords = _missing(channel, session, [None] * count, 'no answer to the respond signal')
    return _assembly(values, coords, None)


def _collect(channel, answers: list):
    """One stored answer from the stored answers of a channel, in the order they arrived.

    Recorded limit: the record and attrs are those of the first answer that has a
    record; collection does not check that all answers' records agree.
    """
    if len(answers) == 1:
        return answers[0]
    parts = [(np.asarray(a.values), _presentation_coords(a)) for a in answers]
    axis = next((list(a['choice'].values) for a in answers if 'choice' in a.dims), None)
    for answer in answers:
        if axis is not None and list(answer['choice'].values) != axis:
            raise ValueError(f'{channel}: the answers have different choice axes ({axis} and '
                             f"{list(answer['choice'].values)}), so they cannot be one assembly; "
                             'each stored answer is in session.emitted')
    names = list(dict.fromkeys(name for _, coords in parts for name in coords))
    merged = {}
    for name in names:
        column = []
        for _, coords in parts:
            n = len(coords['stimulus_id'])
            column += (list(coords[name]) if name in coords else
                       [True] * n if name == 'answer_valid' else
                       [''] * n if name == 'parse_error' else [None] * n)
        merged[name] = column
    if axis is not None:
        merged['choice'] = axis
    first = next((a for a in answers if 'response' in a.attrs), answers[0])
    attrs = {key: value for key, value in first.attrs.items() if key != 'response'}
    return _assembly(np.concatenate([part[0] for part in parts]), merged,
                     first.attrs.get('response'), attrs)
