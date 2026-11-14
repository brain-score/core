"""Behavioral response formats in core: conversions, the response record, storage.

The benchmark specifies the response; the model side produces it; core
validates it; the benchmark scores it (Response Format Standard, section 0).

- **Conversions**: ``argmax`` (probabilities to a choice), ``softmax``
  (continuation log-probabilities to probabilities) and ``softmax_argmax``
  (both in turn). Each returns the validity of every row with its result: a
  row that is not a valid source row gives an invalid result.
- **The response record** (``response_record``): the dict an answer carries
  in ``attrs['response']``. Scores copy the records of their answers.
- **Storage**: the ``behavior`` catalog entry's validator checks a response
  on an addressed ``behavior:<format>`` channel against the session's task and
  the event, and the session stores the checked answer in the schema of its
  format (``_store``). The bare ``behavior`` channel is not checked or stored
  here (legacy handling).

What an addressed response can be: a cell for one trial (``behavior:text``
and ``behavior:choice``: a ``str``; ``behavior:probabilities``: a 1-D array
with one value per option; ``behavior:rating``, ``behavior:confidence``,
``behavior:rt``: a number), or an assembly of rows over ``presentation``
with a ``stimulus_id`` coordinate and a response record. A cell carries no
record; core gives it a native record with ``method='external'`` (for
``behavior:rt`` the units come from ``event.meta['units']``). A malformed
response raises ``ValueError`` and is never stored. The stored answer has
``raw_answer``, ``response_state`` (``valid``, ``invalid``, ``missing``) and
``response_reason`` for every trial, and the record in its attrs. In a session
built from a table, a table row with no answer is stored as a missing trial.
"""
import json
import math
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
    and so is its result; its values are computed as for any row.
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


def _row_options(session, stimulus_id, stimulus_index) -> Optional[list]:
    """The options of one trial, from the task (and the table for per-row options)."""
    from .responses import Choice
    response = _response_of(session)
    if isinstance(response, Choice):
        if not isinstance(response.labels, str):
            return list(response.labels)
        table = getattr(session, 'stimuli', None)
        if table is None:
            return None
        if stimulus_index is not None and stimulus_index in table.index:
            return list(table.loc[stimulus_index, response.labels])
        if stimulus_id is not None and 'stimulus_id' in table.columns:
            rows = table[table['stimulus_id'] == stimulus_id]
            if len(rows):
                return list(rows.iloc[0][response.labels])
        return None
    label_set = getattr(getattr(session, 'task', None), 'label_set', None)
    return list(label_set) if label_set else None


def _axis(session, options) -> Optional[list]:
    """The choice axis: the shared labels, or positions when options differ by row."""
    from .responses import Choice
    response = _response_of(session)
    if options is None:
        return None
    if isinstance(response, Choice) and isinstance(response.labels, str):
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


def _cell_rows(fmt, channel, payload, event, session):
    """A cell as ``(values, coords, record)`` of a one-row stored answer."""
    meta = getattr(event, 'meta', None) or {}
    stimulus_id = meta.get('stimulus_id')
    options = (_row_options(session, stimulus_id, meta.get('stimulus_index'))
               if fmt in ('choice', 'probabilities') else None)
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


def _assembly_rows(fmt, channel, payload, event, session):
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
    meta = getattr(event, 'meta', None) or {}
    if 'stimulus_id' in meta and not isinstance(meta['stimulus_id'], (list, tuple)):
        if ids != [meta['stimulus_id']]:
            raise ValueError(f'{channel}: the event answers stimulus {meta["stimulus_id"]!r}, '
                             f'but its rows are {ids}')
    table = getattr(session, 'stimuli', None)
    if table is not None and 'stimulus_id' in getattr(table, 'columns', ()):
        known = set(table['stimulus_id'])
        unknown = [i for i in ids if i not in known]
        if unknown:
            raise ValueError(f'{channel}: rows {unknown[:5]} are not stimuli of this session')
    if fmt in ('choice', 'probabilities'):
        axis = list(payload['choice'].values)
        options = _row_options(session, ids[0], None) if ids else None
        expected = _axis(session, options)
        if expected is not None and [str(a) for a in axis] != [str(e) for e in expected]:
            raise ValueError(f'{channel}: the choice axis is {axis}, not the trial options {expected}')
    if fmt == 'choice':
        known = np.isnan(values) | (values == 0) | (values == 1)
        if not known.all():
            raise ValueError(f'{channel} values are 0, 1 or NaN (one option, or none)')
        mixed = np.isnan(values).any(axis=1) & ~np.isnan(values).all(axis=1)
        if mixed.any():
            raise ValueError(f'{channel}: a row is all NaN (no choice) or has no NaN')
    if fmt == 'text':
        if any(not (isinstance(v, str) or v is None) for v in values):
            raise ValueError(f'{channel} values are str (or None for no reply)')
    if fmt == 'rt' and record['parameters'].get('units') not in _RT_UNITS:
        raise ValueError(f"{channel}: the record's parameters name units {_RT_UNITS}")
    coords.setdefault('raw_answer', [_raw_text(value) for value in values])
    return values, coords, record


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


def _rows(channel, payload, event, session):
    """The stored ``(values, coords, record, attrs)`` of a response, or ValueError."""
    fmt = _format(channel)
    if hasattr(payload, 'dims') and hasattr(payload, 'attrs'):
        values, coords, record = _assembly_rows(fmt, channel, payload, event, session)
        if fmt in ('choice', 'probabilities'):
            coords['choice'] = list(payload['choice'].values)
        attrs = {key: value for key, value in payload.attrs.items() if key != 'response'}
    else:
        values, coords, record = _cell_rows(fmt, channel, payload, event, session)
        attrs = {}
    if fmt == 'rt' and record['parameters'].get('units') not in _RT_UNITS:
        raise ValueError(f"{channel}: give the units of a reaction time ({_RT_UNITS}): the "
                         "record's parameters['units'], or meta['units'] for one number")
    states, why = _states(fmt, values, coords, record, _response_of(session))
    supplied = coords.pop('response_state', None)
    if supplied is not None and list(supplied) != states:
        raise ValueError(f'{channel}: the supplied states {list(supplied)} contradict the '
                         f'values, which give {states}')
    coords.pop('response_reason', None)
    coords['response_state'], coords['response_reason'] = states, why
    return values, coords, record, attrs


def _validate(payload, event=None, session=None) -> List[str]:
    """The ``behavior`` catalog validator: why an addressed response is malformed."""
    channel = getattr(event, 'channel', None)
    if _format(channel) is None:
        if channel == 'behavior' and _number(payload):
            return ['a number is not a payload of the bare behavior channel']
        return []
    try:
        _rows(channel, payload, event, session)
    except ValueError as error:
        return [str(error)]
    return []


def _store(event, session):
    """The stored answer of a checked response on an addressed behavior channel."""
    if _format(event.channel) is None:
        return event.payload
    values, coords, record, attrs = _rows(event.channel, event.payload, event, session)
    return _assembly(values, coords, record, attrs)


def _table(session):
    table = getattr(session, 'stimuli', None)
    return table if table is not None and 'stimulus_id' in getattr(table, 'columns', ()) else None


def _missing_rows(channel, session, stimulus_ids: list, axis, dtype):
    """Values and coordinates of trials with no answer: NaN (or None), state ``missing``."""
    fmt = _format(channel)
    n = len(stimulus_ids)
    if fmt in ('choice', 'probabilities'):
        if axis is None:
            axis = _axis(session, _row_options(session, stimulus_ids[0] if n else None, None))
        if axis is None:
            raise ValueError(f'{channel}: trials {stimulus_ids[:5]} have no answer and their '
                             'options are not known, so they cannot be stored as missing')
        values = np.full((n, len(axis)), np.nan, dtype=dtype or np.float32)
    elif fmt == 'text':
        values = np.array([None] * n, dtype=object)
    else:
        values = np.full(n, np.nan, dtype=np.float64)
    coords = {'stimulus_id': list(stimulus_ids), 'raw_answer': [None] * n,
              'answer_valid': [False] * n, 'parse_error': [''] * n,
              'response_state': ['missing'] * n, 'response_reason': ['no answer event'] * n}
    return values, coords, axis


def _missing_answer(channel, session, stimulus_ids: list):
    """A stored answer whose trials all had no answer (no record: nothing answered)."""
    values, coords, axis = _missing_rows(channel, session, stimulus_ids, None, None)
    if axis is not None:
        coords['choice'] = axis
    return _assembly(values, coords, None)


def _collect(channel, answers: list, session):
    """One stored answer from the stored answers of a channel's events.

    In a session built from a table, each table row that no answer row names
    is added as a missing trial, and the rows follow the table's order. Raises
    when an answer row cannot be matched to the table.
    """
    table = _table(session)
    if len(answers) == 1 and table is None:
        return answers[0]
    parts = [(np.asarray(a.values), _presentation_coords(a)) for a in answers]
    axis = list(answers[0]['choice'].values) if answers and 'choice' in answers[0].dims else None
    for answer in answers[1:]:
        if axis is not None and list(answer['choice'].values) != axis:
            raise ValueError(f'{channel}: the answers have different choice axes')
    if table is not None:
        wanted = list(table['stimulus_id'])
        remaining = list(wanted)
        for _, coords in parts:
            for stimulus_id in coords['stimulus_id']:
                if _null(stimulus_id):
                    raise ValueError(f'{channel}: an answer has no stimulus_id, so the trials '
                                     'without an answer cannot be found')
                if stimulus_id not in remaining:
                    raise ValueError(f'{channel}: stimulus {stimulus_id!r} is answered more '
                                     'often than it is in the table')
                remaining.remove(stimulus_id)
        if remaining:
            dtype = parts[0][0].dtype if parts else None
            values, coords, axis = _missing_rows(channel, session, remaining, axis, dtype)
            parts.append((values, coords))
        elif len(answers) == 1:
            return answers[0]
    if not parts:
        raise ValueError(f'No emitted events for channel {channel!r}.')
    names = list(dict.fromkeys(name for _, coords in parts for name in coords))
    merged = {}
    for name in names:
        column = []
        for _, coords in parts:
            n = len(coords['stimulus_id'])
            if name in coords:
                column += list(coords[name])
            elif table is not None and name in table.columns:
                column += [table.loc[table['stimulus_id'] == i, name].iloc[0]
                           for i in coords['stimulus_id']]
            else:
                column += ([True] * n if name == 'answer_valid' else
                           [''] * n if name == 'parse_error' else [None] * n)
        merged[name] = column
    values = np.concatenate([part[0] for part in parts])
    if table is not None:
        wanted = list(table['stimulus_id'])
        order = sorted(range(len(values)), key=lambda i: wanted.index(merged['stimulus_id'][i]))
        values = values[order]
        merged = {name: [column[i] for i in order] for name, column in merged.items()}
    if axis is not None:
        merged['choice'] = axis
    record = answers[0].attrs.get('response') if answers else None
    attrs = {key: value for key, value in answers[0].attrs.items() if key != 'response'} \
        if answers else {}
    return _assembly(values, merged, record, attrs)
