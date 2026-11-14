"""The behavioral response formats: conversions and the response record.

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
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
