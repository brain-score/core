"""The behavioral response formats: conversions, the response record, validators.

The benchmark specifies the response; the model side produces it; core
validates it; the benchmark scores it (Response Format Standard, section 0).
This module is core's part:

- **Conversions** between formats, stated by a registration and recorded:
  ``argmax`` (probabilities to a choice), ``softmax`` (continuation
  log-probabilities to probabilities) and ``softmax_argmax`` (both in turn).
  Every caller uses these functions, so one algorithm gives one number.
- **The response record** (``response_record``): one dict in the ``response``
  attribute of every answer, copied to every score. It says which format the
  answer is in, whether the model gave it natively or by a conversion, and how.
- **Validation** (``PAYLOAD_VALIDATORS``): the channel catalog runs the
  validator of a response's format when a session accepts it. A malformed
  payload (wrong type, number of dimensions, shape or coordinate layout, or a
  malformed record) is a defect of the model side and raises there; it is
  never kept as an answer.
- **Trial states** (``assign_states``): a well-formed answer keeps every
  trial, each with a state (``valid``, ``invalid`` or ``missing``), its raw
  response and a reason. An invalid or missing trial is not an error.

The tolerance on the sum of a probability row is ``PROBABILITY_TOLERANCE``,
defined here only.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# A row of probabilities is valid when it sums to 1 within this absolute
# tolerance. Readout probabilities are stored as float32, whose rounding error
# over a few options is about 1e-7.
PROBABILITY_TOLERANCE = 1e-5

VALID, INVALID, MISSING = 'valid', 'invalid', 'missing'
STATES = (VALID, INVALID, MISSING)

CONVERSIONS = ('argmax', 'softmax', 'softmax_argmax')
CATEGORIES = ('behavioral', 'model-derived')
ORIGINS = ('native', 'derived')
PARSERS = ('model', 'library')
RECORD_FIELDS = ('format', 'category', 'origin', 'method', 'conversion', 'source_format',
                 'parameters', 'parsed_by')


# --------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------

def argmax(probabilities) -> Tuple[np.ndarray, np.ndarray]:
    """The most probable option of each row: ``(indices, valid)``.

    Ties go to the first maximum in the given column order (numpy ``argmax``).
    The column order is the caller's: pass the columns in the trial's option
    order and record it. A row with a non-finite value has no choice: its index
    is -1 and ``valid`` is False. The result is the most probable option under
    the model's distribution, not a sampled choice.
    """
    values = np.asarray(probabilities)
    if values.ndim != 2:
        raise ValueError(f'argmax takes a (trials, options) array; got shape {values.shape}')
    valid = np.isfinite(values).all(axis=1)
    indices = np.where(valid, values.argmax(axis=1), -1)
    return indices, valid


def softmax(log_probabilities) -> np.ndarray:
    """Each row of log-probabilities as probabilities over the row's options.

    Temperature 1, in the given column order. The row's offset is lost: two rows
    that differ by a constant give the same probabilities. The result is a
    distribution over the listed continuations only. A row that holds NaN or
    +inf, or only -inf, gives a row of NaN, which ``argmax`` treats as invalid.
    """
    logp = np.asarray(log_probabilities)
    logp = logp - logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def softmax_argmax(log_probabilities) -> Tuple[np.ndarray, np.ndarray]:
    """``argmax(softmax(log_probabilities))``: the most likely listed continuation."""
    return argmax(softmax(log_probabilities))


_DEFAULT_PARAMETERS = {
    'argmax': {'tie': 'first', 'option_order': 'trial'},
    'softmax': {'temperature': 1, 'option_order': 'trial'},
    'softmax_argmax': {'temperature': 1, 'tie': 'first', 'option_order': 'trial'},
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
    a ``BrainScoreModel`` names its own). A conversion makes the answer
    ``derived`` and needs ``source_format``; its ``parameters`` default to the
    conversion's stated ones. A derived answer without a conversion is a link
    (``method='link'``). ``parsed_by`` is ``'model'`` when the model side parsed
    generated text, ``'library'`` when an external library parses it.
    """
    if conversion is not None and conversion not in CONVERSIONS:
        raise ValueError(f'conversion is one of {CONVERSIONS} or None, not {conversion!r}')
    if category not in CATEGORIES:
        raise ValueError(f'category is one of {CATEGORIES}, not {category!r}')
    if parsed_by not in (None, *PARSERS):
        raise ValueError(f'parsed_by is one of {PARSERS} or None, not {parsed_by!r}')
    if not isinstance(method, str) or not method:
        raise ValueError(f'method is a nonempty string, not {method!r}')
    derived = conversion is not None or method == 'link'
    if derived and not source_format:
        raise ValueError('A derived answer names its source_format')
    if parameters is None:
        parameters = dict(_DEFAULT_PARAMETERS.get(conversion, {}))
    return {'format': format, 'category': category, 'origin': 'derived' if derived else 'native',
            'method': method, 'conversion': conversion, 'source_format': source_format,
            'parameters': dict(parameters), 'parsed_by': parsed_by}


def record_problems(record, channel: str) -> List[str]:
    """Why ``record`` is not a well-formed response record for ``channel``; empty when it is."""
    if not isinstance(record, dict):
        return [f'the response record is {type(record).__name__}, not a dict']
    if tuple(record) != RECORD_FIELDS:
        return [f'the response record has fields {tuple(record)}, not {RECORD_FIELDS}']
    problems = []
    if record['format'] != channel:
        problems.append(f"the response record names format {record['format']!r}, "
                        f'but the answer is on {channel!r}')
    if record['category'] not in CATEGORIES:
        problems.append(f"the record's category {record['category']!r} is not one of {CATEGORIES}")
    if record['origin'] not in ORIGINS:
        problems.append(f"the record's origin {record['origin']!r} is not one of {ORIGINS}")
    if record['conversion'] not in (None, *CONVERSIONS):
        problems.append(f"the record's conversion {record['conversion']!r} is not one of {CONVERSIONS}")
    if record['origin'] == 'derived' and not record['source_format']:
        problems.append('a derived answer names its source_format')
    if not isinstance(record['parameters'], dict):
        problems.append("the record's parameters are not a dict")
    return problems


# --------------------------------------------------------------------------
# Validators: a malformed payload raises
# --------------------------------------------------------------------------

def _is_assembly(payload) -> bool:
    return hasattr(payload, 'dims') and hasattr(payload, 'attrs')


def _assembly_problems(payload, channel: str, dims: Tuple[str, ...], numeric: bool) -> List[str]:
    if tuple(payload.dims) != dims:
        return [f'payload dims are {tuple(payload.dims)}, not {dims}']
    if dims == ('presentation', 'choice') and payload.sizes['choice'] < 1:
        return ['payload has no options on its choice axis']
    kind = np.asarray(payload.values).dtype.kind
    if numeric and kind not in 'fiub':
        return [f'payload values are {np.asarray(payload.values).dtype}, not numbers']
    if not numeric and kind != 'O':
        return [f'payload values are {np.asarray(payload.values).dtype}, not Python objects (str)']
    if 'response' in payload.attrs:
        return record_problems(payload.attrs['response'], channel)
    return []


def _array_problems(payload, ndim: Tuple[int, ...]) -> List[str]:
    if payload.ndim not in ndim:
        return [f'payload has {payload.ndim} dimensions, not {" or ".join(map(str, ndim))}']
    if payload.dtype.kind not in 'fiub':
        return [f'payload values are {payload.dtype}, not numbers']
    return []


def _validator(channel: str, dims: Tuple[str, ...], numeric: bool, ndim: Tuple[int, ...],
               label_allowed: bool):
    def problems(payload) -> List[str]:
        if _is_assembly(payload):
            return _assembly_problems(payload, channel, dims, numeric)
        if isinstance(payload, str):
            return [] if label_allowed else [f'payload is a str; {channel} takes numbers']
        if isinstance(payload, np.ndarray):
            return _array_problems(payload, ndim) if numeric else [
                f'payload is an array; {channel} takes text']
        return []   # the family's payload type check reports any other type
    return problems


_ROWS = ('presentation', 'choice')
_ONE = ('presentation',)
# The validator of each format address. The catalog runs it when a session
# accepts a response on that address (``io_catalog.validate``).
PAYLOAD_VALIDATORS = {
    'behavior:choice': _validator('behavior:choice', _ROWS, True, (1,), label_allowed=True),
    'behavior:probabilities': _validator('behavior:probabilities', _ROWS, True, (1,), False),
    'behavior:rating': _validator('behavior:rating', _ONE, True, (0, 1), False),
    'behavior:confidence': _validator('behavior:confidence', _ONE, True, (0, 1), False),
    'behavior:rt': _validator('behavior:rt', _ONE, True, (0, 1), False),
    'behavior:text': _validator('behavior:text', _ONE, False, (), label_allowed=True),
}


# --------------------------------------------------------------------------
# Trial states: an invalid or missing trial is kept
# --------------------------------------------------------------------------

def _is_null(value) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _value_reasons(channel: str, values: np.ndarray, record: Optional[dict],
                   response) -> List[Optional[str]]:
    """Why each trial's value is not a usable answer (None when it is)."""
    reasons: List[Optional[str]] = []
    if channel == 'behavior:choice':
        for row in values:
            if not np.isfinite(row).all():
                reasons.append('no option chosen')
            elif not (np.isin(row, (0, 1)).all() and (row == 1).sum() == 1):
                reasons.append('not exactly one option chosen')
            else:
                reasons.append(None)
    elif channel == 'behavior:probabilities':
        log_scale = (record or {}).get('category') == 'model-derived'
        for row in values:
            if not np.isfinite(row).all():
                reasons.append('a value is not finite')
            elif log_scale:
                reasons.append('a log probability is above 0' if (row > 0).any() else None)
            elif (row < 0).any():
                reasons.append('a probability is negative')
            elif abs(float(row.sum()) - 1.0) > PROBABILITY_TOLERANCE:
                reasons.append(f'the probabilities sum to {float(row.sum())!r}, not 1 '
                               f'within {PROBABILITY_TOLERANCE}')
            else:
                reasons.append(None)
    elif channel in ('behavior:rating', 'behavior:confidence'):
        scale = getattr(response, 'confidence', None) if channel == 'behavior:confidence' else response
        low, high = getattr(scale, 'low', None), getattr(scale, 'high', None)
        for value in values:
            value = float(value)
            if not np.isfinite(value):
                reasons.append('no number')
            elif low is not None and not low <= value <= high:
                reasons.append(f'{value} is outside {low} to {high}')
            else:
                reasons.append(None)
    elif channel == 'behavior:rt':
        for value in values:
            value = float(value)
            reasons.append('not finite' if not np.isfinite(value)
                           else 'negative' if value < 0 else None)
    elif channel == 'behavior:text':
        reasons = [None if isinstance(value, str) else 'not a string' for value in values]
    return reasons


def _coordinate(payload, name: str, default: list) -> list:
    # A level of the presentation index is not listed in ``coords`` (xarray
    # 2022.3), so look it up instead of testing membership.
    try:
        return list(payload[name].values)
    except KeyError:
        return default


def assign_states(payload, channel: str, response=None):
    """``payload`` with each trial's state and reason, as core assigns them.

    Only a format answer that is an assembly over ``presentation`` gets states;
    any other payload is returned as it is. ``response`` is the task's
    ``Choice``, ``Scale`` or ``Text`` (the specification), when known.

    A trial is ``missing`` when its raw response (``raw_answer``) is None: the
    subject gave no answer. It is ``invalid`` when the model side marked it so
    (``answer_valid`` False, with its ``parse_error``) or its value breaks the
    format's rules (``_value_reasons``). Otherwise it is ``valid``. The values
    are not changed. The states go in ``response_state`` and the reasons in
    ``response_reason`` (empty for a valid trial).
    """
    if channel not in PAYLOAD_VALIDATORS or not _is_assembly(payload) \
            or 'presentation' not in payload.dims:
        return payload
    values = np.asarray(payload.values)
    if payload.dims[0] != 'presentation':
        return payload
    record = payload.attrs.get('response')
    reasons = _value_reasons(channel, values, record, response)
    n = len(reasons)
    raw = _coordinate(payload, 'raw_answer', [''] * n)
    marked = _coordinate(payload, 'answer_valid', [True] * n)
    errors = _coordinate(payload, 'parse_error', [''] * n)
    states, why = [], []
    for value_reason, raw_answer, ok, error in zip(reasons, raw, marked, errors):
        if _is_null(raw_answer):
            states.append(MISSING)
            why.append('no answer')
        elif not ok or value_reason is not None:
            states.append(INVALID)
            why.append(str(error) if error else value_reason or 'marked invalid by the model side')
        else:
            states.append(VALID)
            why.append('')
    return payload.assign_coords(response_state=('presentation', states),
                                 response_reason=('presentation', why))
