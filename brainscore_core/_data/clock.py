"""Optional clock descriptor for time-resolved payloads.

A clock descriptor says what the numbers on a ``time_bin`` axis mean. It is a
plain dict, stored at ``assembly.attrs['temporal_clock']`` (or under another
name, for example ``'scanner_clock'`` for scanner samples). Nothing in core
writes or requires it: producers that know their clock add it, and consumers
that need it check it before they align or merge time axes.

Keys:

- ``units`` (required): ``'ms'``.
- ``origin`` (required): where time zero is. One of :data:`ORIGINS`.
- ``reference_input`` (required when ``origin == 'media_start'``): the input
  channel whose media defines time zero, for example ``'vision:clip'``.
- ``anchor`` (required): which point of a sample the timestamps name. One of
  :data:`ANCHORS`: ``'start'`` (intervals, start and end are primary),
  ``'center'`` (centers are primary, edges may be derived) or ``'instant'``
  (point samples).
- ``interval`` (optional, default ``'half_open'``): bins are ``[start, end)``.
- ``accuracy`` (required): how the timestamps are known. One of
  :data:`ACCURACIES`. Accuracy is a claim made by the producer; it is not
  inferred here.
- ``description`` (required): the evidence or source of the timestamps, as
  free text (for example "decoded-frame PTS" or "nominal TR from BIDS").
- ``tolerance_ms`` (optional): the stated uncertainty of each timestamp, in
  ms, finite and not negative.
- ``coverage_ms`` / ``missing_ms`` (optional): sorted, non-overlapping
  ``[start, end)`` intervals with real data / without real data (for example
  padding past the media end). A coverage interval must not overlap a
  missing interval.
- ``reference_sha256`` (optional): the SHA-256 of the reference media file.
- ``media_offset_ms`` (optional): where the reference media starts on the
  experiment's clock, in ms. It is recorded and validated, not compared.

All checks return data. :func:`validate_clock` and :func:`validate_time_axis`
return a list of problems (empty when valid). :func:`check_clock_compatibility`
returns a :class:`ClockMatch`. Only :func:`clock_descriptor`, an explicit
builder, raises on an invalid descriptor. Clock compatibility is separate
from channel compatibility; pre-flight does not call it.

Migration note: clock dicts written before this module (for example the
``temporal_clock`` and ``scanner_clock`` dicts of the CNeuroMod branch) have
no ``anchor`` key and must add one to pass :func:`validate_clock`. The anchor
is a claim about the data and must come from evidence. For scanner samples,
``arange(n) * TR`` alone does not establish whether a time names the volume
start, the slice-timing reference or another point; state what the
acquisition evidence supports and lower ``accuracy`` when it is not known.
"""

import math
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

CLOCK_ATTR = 'temporal_clock'
UNITS = ('ms',)
ORIGINS = ('media_start', 'clip_start', 'trial_onset', 'run_start', 'session_start')
ANCHORS = ('start', 'center', 'instant')
INTERVALS = ('half_open',)
ACCURACIES = ('exact', 'measured', 'verified', 'nominal', 'approximate', 'unknown')

_REQUIRED = ('units', 'origin', 'anchor', 'accuracy', 'description')
_OPTIONAL = ('reference_input', 'interval', 'tolerance_ms', 'coverage_ms', 'missing_ms',
             'reference_sha256', 'media_offset_ms')
# Same slack as the CNeuroMod/Lahner temporal_aligned.validate_stream end check.
DURATION_TOLERANCE_MS = 1e-6
_FIELDS = _REQUIRED + _OPTIONAL


@dataclass(frozen=True)
class ClockMatch:
    """Result of :func:`check_clock_compatibility`. True when compatible."""
    compatible: bool
    reasons: Tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.compatible


def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _interval_problems(name: str, intervals: Any) -> List[str]:
    if isinstance(intervals, (str, bytes)) or not isinstance(intervals, Sequence):
        return [f'{name} must be a sequence of [start, end) pairs']
    problems, previous_end = [], None
    for pair in intervals:
        if (isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2
                or not all(_is_number(value) for value in pair)):
            problems.append(f'{name} has a malformed interval: {pair!r}')
            continue
        start, end = pair
        if end <= start:
            problems.append(f'{name} has an empty or reversed interval: {pair!r}')
        elif previous_end is not None and start < previous_end:
            problems.append(f'{name} intervals are unsorted or overlap at {pair!r}')
        previous_end = end if previous_end is None else max(previous_end, end)
    return problems


def _well_formed_pairs(intervals: Any) -> List[Tuple[float, float]]:
    if isinstance(intervals, (str, bytes)) or not isinstance(intervals, Sequence):
        return []
    return [tuple(pair) for pair in intervals
            if not isinstance(pair, (str, bytes)) and isinstance(pair, Sequence)
            and len(pair) == 2 and all(_is_number(value) for value in pair)
            and pair[1] > pair[0]]


def validate_clock(clock: Any) -> List[str]:
    """Problems with a clock descriptor; an empty list means it is valid."""
    if not isinstance(clock, Mapping):
        return ['clock must be a mapping']
    problems = [f'missing {key}' for key in _REQUIRED if key not in clock]
    problems += [f'unknown key {key!r}' for key in clock if key not in _FIELDS]
    for key, allowed in (('units', UNITS), ('origin', ORIGINS), ('anchor', ANCHORS),
                         ('accuracy', ACCURACIES), ('interval', INTERVALS)):
        if key in clock and clock[key] not in allowed:
            problems.append(f'{key} {clock[key]!r} is not one of {allowed}')
    if 'description' in clock and not (isinstance(clock['description'], str)
                                       and clock['description'].strip()):
        problems.append('description must state the evidence for the timestamps')
    reference = clock.get('reference_input')
    if reference is not None and not (isinstance(reference, str) and reference.strip()):
        problems.append('reference_input must be a channel name')
    if clock.get('origin') == 'media_start' and reference is None:
        problems.append('a media_start clock must name its reference_input')
    if 'tolerance_ms' in clock and not (_is_number(clock['tolerance_ms'])
                                        and clock['tolerance_ms'] >= 0):
        problems.append('tolerance_ms must be a finite number >= 0')
    for key in ('coverage_ms', 'missing_ms'):
        if key in clock:
            problems += _interval_problems(key, clock[key])
    covered = _well_formed_pairs(clock.get('coverage_ms', ()))
    for start, end in _well_formed_pairs(clock.get('missing_ms', ())):
        if any(start < c_end and c_start < end for c_start, c_end in covered):
            problems.append(f'missing_ms interval [{start}, {end}) overlaps coverage_ms')
    sha = clock.get('reference_sha256')
    if 'reference_sha256' in clock and not (isinstance(sha, str) and sha.strip()):
        problems.append('reference_sha256 must be a non-empty string')
    if 'media_offset_ms' in clock and not _is_number(clock['media_offset_ms']):
        problems.append('media_offset_ms must be a finite number')
    return problems


def clock_descriptor(*, units: str = 'ms', origin: str, anchor: str, accuracy: str,
                     description: str, reference_input: Optional[str] = None,
                     interval: str = 'half_open', tolerance_ms: Optional[float] = None,
                     coverage_ms: Optional[Sequence[Tuple[float, float]]] = None,
                     missing_ms: Optional[Sequence[Tuple[float, float]]] = None,
                     reference_sha256: Optional[str] = None,
                     media_offset_ms: Optional[float] = None) -> dict:
    """Build a valid clock descriptor dict. Raises ``ValueError`` if invalid."""
    clock = {'units': units, 'origin': origin, 'anchor': anchor, 'interval': interval,
             'accuracy': accuracy, 'description': description}
    for key, value in (('reference_input', reference_input), ('tolerance_ms', tolerance_ms),
                       ('reference_sha256', reference_sha256),
                       ('media_offset_ms', media_offset_ms)):
        if value is not None:
            clock[key] = value
    for key, value in (('coverage_ms', coverage_ms), ('missing_ms', missing_ms)):
        if value is not None:
            clock[key] = [list(pair) for pair in value]
    problems = validate_clock(clock)
    if problems:
        raise ValueError('Invalid clock descriptor: ' + '; '.join(problems))
    return clock


def check_clock_compatibility(a: Mapping, b: Mapping, *,
                              tolerance_ms: Optional[float] = None,
                              strict: bool = False) -> ClockMatch:
    """Structural compatibility of two declared clocks.

    Compatible means both descriptors are valid and declare the same units,
    origin, reference input, anchor and interval convention, and do not give
    different ``reference_sha256`` values. It is a check of the declarations,
    not proof that the timing of the data is identical. The evidence text and
    ``media_offset_ms`` are not compared.

    With ``tolerance_ms`` (finite, >= 0), neither clock may have
    ``accuracy='unknown'``, both must state ``tolerance_ms``, and their sum
    must not exceed it. Returns a :class:`ClockMatch` with the reasons for any
    mismatch; with ``strict=True`` a mismatch raises ``ValueError`` instead.
    """
    reasons: List[str] = []
    for label, clock in (('first', a), ('second', b)):
        reasons += [f'{label} clock: {problem}' for problem in validate_clock(clock)]
    if tolerance_ms is not None and not (_is_number(tolerance_ms) and tolerance_ms >= 0):
        reasons.append(f'requested tolerance_ms {tolerance_ms!r} is not a finite number >= 0')
        tolerance_ms = None
    if not reasons:
        for key in ('units', 'origin', 'anchor'):
            if a[key] != b[key]:
                reasons.append(f'{key} differs: {a[key]!r} vs {b[key]!r}')
        if a.get('interval', 'half_open') != b.get('interval', 'half_open'):
            reasons.append('interval convention differs')
        if a.get('reference_input') != b.get('reference_input'):
            reasons.append(f"reference_input differs: {a.get('reference_input')!r} "
                           f"vs {b.get('reference_input')!r}")
        shas = (a.get('reference_sha256'), b.get('reference_sha256'))
        if None not in shas and shas[0] != shas[1]:
            reasons.append(f'reference_sha256 differs: {shas[0]!r} vs {shas[1]!r}')
        if tolerance_ms is not None:
            stated = [clock.get('tolerance_ms') for clock in (a, b)]
            if 'unknown' in (a['accuracy'], b['accuracy']):
                reasons.append(f'accuracy cannot be checked against {tolerance_ms} ms: '
                               "a clock has accuracy 'unknown'")
            elif None in stated:
                reasons.append(f'accuracy cannot be checked against {tolerance_ms} ms: '
                               'a clock states no tolerance_ms')
            elif sum(stated) > tolerance_ms:
                reasons.append(f'accuracy beyond tolerance: {stated[0]} + {stated[1]} ms '
                               f'> {tolerance_ms} ms')
    match = ClockMatch(compatible=not reasons, reasons=tuple(reasons))
    if strict and not match:
        raise ValueError('Incompatible clocks: ' + '; '.join(match.reasons))
    return match


class _NonNumeric(Exception):
    pass


def _time_coord(assembly, name: str):
    """Float values of a ``time_bin`` coordinate (MultiIndex levels too), or None."""
    import numpy as np
    try:
        coord = assembly[name]
    except KeyError:
        return None
    if tuple(coord.dims) != ('time_bin',):
        return None
    try:
        values = np.asarray(coord.values, dtype=float)
    except (TypeError, ValueError):
        raise _NonNumeric(name)
    return values


def validate_time_axis(assembly, clock: Optional[Mapping] = None,
                       duration_ms: Optional[float] = None) -> List[str]:
    """Problems with an assembly's ``time_bin`` coordinates under its clock.

    ``clock`` defaults to ``assembly.attrs['temporal_clock']``. The axis must
    not be empty and every time must be numeric and finite.

    Start and center anchors follow ``temporal_aligned.validate_stream``:
    ``time_bin_start_ms`` and ``time_bin_end_ms`` are required; centers are
    ``time_bin_center_ms`` if present, else ``(start + end) / 2``; ends must be
    after starts; starts, ends and centers must each strictly increase; each
    center must lie in ``[start, end)``. Instant anchors need strictly
    increasing sample times (``time_bin_center_ms``, else
    ``time_bin_start_ms``).

    With ``duration_ms``, no time may be negative and none may exceed
    ``duration_ms + DURATION_TOLERANCE_MS``, as in ``validate_stream``.
    Without it, negative times are allowed (for example before a trial onset).
    """
    import numpy as np
    if clock is None:
        clock = getattr(assembly, 'attrs', {}).get(CLOCK_ATTR)
        if clock is None:
            return [f'no clock given and no attrs[{CLOCK_ATTR!r}]']
    problems = validate_clock(clock)
    if problems:
        return problems
    if 'time_bin' not in assembly.dims:
        return ['assembly has no time_bin dimension']
    if duration_ms is not None and not (_is_number(duration_ms) and duration_ms > 0):
        return ['duration_ms must be a finite number > 0']
    try:
        starts = _time_coord(assembly, 'time_bin_start_ms')
        ends = _time_coord(assembly, 'time_bin_end_ms')
        centers = _time_coord(assembly, 'time_bin_center_ms')
    except _NonNumeric as error:
        return [f'{error.args[0]} is not numeric']
    if clock['anchor'] == 'instant':
        times = centers if centers is not None else starts
        if times is None:
            return ['instant clock needs time_bin_center_ms or time_bin_start_ms']
        checked = [times]
    else:
        if starts is None or ends is None:
            return ['interval clock needs time_bin_start_ms and time_bin_end_ms on time_bin']
        if centers is None:
            centers = (starts + ends) / 2
        checked = [starts, ends, centers]
    if not len(checked[0]):
        return ['empty time axis']
    if not all(np.isfinite(values).all() for values in checked):
        return ['non-finite timestamps']
    if clock['anchor'] == 'instant':
        if np.any(np.diff(times) <= 0):
            problems.append('timestamps are not strictly increasing')
    else:
        if np.any(ends <= starts):
            problems.append('a bin ends at or before its start')
        if any(np.any(np.diff(values) <= 0) for values in checked):
            problems.append('starts, ends or centers are not strictly increasing')
        if np.any(centers < starts) or np.any(centers >= ends):
            problems.append('a center lies outside its [start, end) bin')
    if duration_ms is not None:
        if any(np.any(values < 0) for values in checked):
            problems.append('timestamps before the origin')
        if any(np.any(values > duration_ms + DURATION_TOLERANCE_MS) for values in checked):
            problems.append(f'timestamps past the duration ({duration_ms} ms)')
    return problems


__all__ = [
    'CLOCK_ATTR', 'DURATION_TOLERANCE_MS', 'UNITS', 'ORIGINS', 'ANCHORS', 'INTERVALS', 'ACCURACIES',
    'ClockMatch', 'validate_clock', 'clock_descriptor', 'check_clock_compatibility',
    'validate_time_axis',
]
