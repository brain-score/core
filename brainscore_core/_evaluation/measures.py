"""Record how the behavioral responses of one scoring run were given.

A behavioral answer carries its response record in ``attrs['response']``
(``brainscore_core.response_formats.response_record``). ``score()`` opens
``record_measures()`` around the benchmark run. Core notes the record where a
session accepts a response from the subject (``note_response`` in each core
session's ``respond`` / ``emit``), event by event, before any session
concatenates payloads. Events that no session accepts (saved records,
fixtures, copies) are not noted. The score metadata then copies the records
to the score. Only returned data is read, never the subject's state.

The same record keeps the execution configuration of the run's sessions, noted
where a session accepts a response (``note_session``): the delivery form
(``'listed'``: one list known up front; ``'stream'``: one event at a time) and
the regions-per-pass plan a subject says it used (``note_session(session,
regions_per_pass)``). Core does not read the plan itself: only the subject that
owns it knows whether it honors it, so a subject that passes none is stamped
None. Either can change the last digits of a score, never its value otherwise.

The same note carries the provenance of a session (rule 6.3), once per session:
``note_session(session, provenance=entry)``, where the subject that owns its
parts says which parts the session used and their facts (``BrainScoreModel``
lists each extractor and answering part; a registration adds its own facts as
a separate entry). Core adds the activation store state of the session (the
``RESULTCACHING_DISABLE`` value at that moment) and keeps the distinct entries.
No response, payload or event meta is read or changed for it.

ponytail: the record is a ``ContextVar``. A response made in a worker thread or
process is not noted. No behavioral path in core, vision, language or unified
makes its response events there today (worker pools only extract features or
compute metrics); copy the context into the worker if one ever does.
"""

import itertools
import json
import os
from copy import deepcopy
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Iterator, Optional, Set, Tuple

_active: ContextVar[Optional['MeasureRecord']] = ContextVar('measure_record', default=None)


class MeasureRecord:
    """The response records seen during one scoring run, and how often its
    sessions applied and undid each perturbation handle (``perturbation``
    responses carry a ``handle_id``; ``meta['kind'] == 'reset'`` marks an undo).
    Counts, not sets: a handle name used again after its undo is counted again."""

    def __init__(self) -> None:
        self._records: Dict[str, dict] = {}   # one per distinct record, keyed by its JSON
        self.applied: Counter = Counter()
        self.undone: Counter = Counter()
        self._deliveries: Set[str] = set()
        self._region_plans: Set[str] = set()
        self._provenance: Dict[str, dict] = {}   # one per distinct session entry, keyed by its JSON
        self._store_states: Set[str] = set()
        self._sessions: Dict[int, Optional[str]] = {}   # session key -> the subject that took part, if named
        self._stated: Set[str] = set()                  # subjects that stated their parts

    @property
    def unreleased(self) -> Set[str]:
        """Handles applied during the run more often than the run undid them."""
        return set(self.applied - self.undone)

    def check_released(self, benchmark_identifier: str, subject_identifier: str) -> None:
        """Raise when the run left an intervention it applied active.

        The one release check of every score entry point: the run must leave
        the subject as it found it. Interventions the caller applied stay.
        """
        if self.unreleased:
            raise RuntimeError(
                f"Benchmark '{benchmark_identifier}' left intervention handle(s) "
                f"{sorted(self.unreleased)} active on subject '{subject_identifier}'; a "
                "benchmark must undo every intervention it applies: apply_state_change("
                "subject, StateChange(kind='reset', handle_id=...)).")

    @property
    def responses(self) -> Tuple[dict, ...]:
        """The distinct response records of the run, sorted by format, then content."""
        return tuple(self._records[key] for key in sorted(
            self._records, key=lambda key: (self._records[key]['format'], key)))

    @property
    def provenance(self) -> Tuple[dict, ...]:
        """The distinct provenance entries the run's sessions noted, sorted by content."""
        return tuple(self._provenance[key] for key in sorted(self._provenance))


@contextmanager
def record_measures() -> Iterator[MeasureRecord]:
    """Collect the measure of each behavioral response until the block ends.

    A record opened inside another one (a benchmark that calls ``score()``)
    passes its measures to the outer record when it ends without an error, so
    the outer score describes the responses its result came from. A failed
    inner run passes nothing on: its result does not reach the outer score.
    """
    outer = _active.get()
    record = MeasureRecord()
    token = _active.set(record)
    succeeded = False
    try:
        yield record
        succeeded = True
    finally:
        _active.reset(token)
        if succeeded and outer is not None:
            outer._records.update(record._records)
            outer.applied += record.applied
            outer.undone += record.undone
            outer._deliveries |= record._deliveries
            outer._region_plans |= record._region_plans
            outer._provenance.update(record._provenance)
            outer._store_states |= record._store_states
            outer._sessions.update(record._sessions)
            outer._stated |= record._stated


def _note_record(payload) -> None:
    """Add ``payload.attrs['response']`` to the open record, if any."""
    record = _active.get()
    if record is None:
        return
    attrs = getattr(payload, 'attrs', None)
    response = attrs.get('response') if isinstance(attrs, dict) else None
    if isinstance(response, dict) and 'format' in response:
        record._records[json.dumps(response, sort_keys=True, default=str)] = response


def _store_state() -> str:
    """The activation store state now: ``'on'``, ``'off'``, or ``'off for: <prefixes>'``.

    The rule of ``result_caching.is_enabled``: ``'1'`` disables every function,
    an empty value or ``'0'`` none, any other value the listed identifier prefixes.
    """
    value = os.environ.get('RESULTCACHING_DISABLE', '0')
    if value == '1':
        return 'off'
    return 'on' if value in ('', '0') else f'off for: {value}'


def note_session(session, regions_per_pass: Optional[str] = None,
                 provenance: Optional[dict] = None, subject: Optional[str] = None) -> None:
    """Note how ``session`` delivers its inputs and, when given, the plan the subject used,
    the provenance of the session and the subject that takes part in it.

    ``regions_per_pass`` comes from the subject that owns the setting and validated
    it (``BrainScoreModel``); a subject that does not read it passes none.
    ``provenance`` is one entry (a mapping of plain values with a ``subject``) from
    the subject that owns the parts; core adds ``activation_store`` (the store state
    of the session) and keeps a copy. An entry with ``parts`` says that its subject
    stated its parts. ``subject`` names the subject that takes part in ``session``,
    for a caller that runs several subjects (a conversation harness); a session that
    no note names belongs to the scored subject. Each subject that took part and
    stated no parts gets an entry with its facts ``'not stated'`` (``provenance_attrs``).
    """
    record = _active.get()
    if record is None:
        return
    record._deliveries.add('stream' if session.streaming else 'listed')
    record._store_states.add(_store_state())
    if regions_per_pass is not None:
        record._region_plans.add(regions_per_pass)
    key = _session_key(session)
    named = subject if subject is not None else (provenance or {}).get('subject')
    if named is not None or key not in record._sessions:
        record._sessions[key] = named if named is not None else record._sessions.get(key)
    if provenance is not None:
        entry = {**deepcopy(dict(provenance)), 'activation_store': _store_state()}
        record._provenance[json.dumps(entry, sort_keys=True, default=str)] = entry
        if 'parts' in entry:
            record._stated.add(entry.get('subject'))


_SESSION_KEYS = itertools.count()


def _session_key(session) -> int:
    """A number for ``session`` that lasts as long as the session: the record keeps no session alive."""
    key = getattr(session, '_measure_key', None)
    if key is None:
        key = next(_SESSION_KEYS)
        try:
            session._measure_key = key
        except AttributeError:
            key = id(session)
    return key


def execution_attrs(record: MeasureRecord) -> dict:
    """``{'delivery': ..., 'regions_per_pass': ...}`` of a run, for the score's attrs.

    Each value is one string, or a sorted tuple when the sessions differed.
    ``delivery`` is None when no session answered; ``regions_per_pass`` is None
    when no subject said which plan it used.
    """
    def one(values, default):
        values = sorted(values)
        return default if not values else values[0] if len(values) == 1 else tuple(values)
    return {'delivery': one(record._deliveries, None),
            'regions_per_pass': one(record._region_plans, None)}


NOT_STATED = 'not stated'


def provenance_attrs(record: Optional[MeasureRecord], subject_identifier) -> Tuple[dict, ...]:
    """``attrs['provenance']`` of a score: the noted entries, and one entry per silent participant.

    A subject that took part in a session of the score and stated no parts (an old
    model through an adapter, a custom subject, a subject beside a model in a
    conversation) gets ``{'subject': ..., 'parts': 'not stated', 'activation_store': ...}``
    beside the entries of the others. A session that no note names belongs to the
    scored subject (``subject_identifier``); with no session at all, the scored subject
    is the one participant. The store state is the run's (``'not stated'`` when no
    session answered). Entries are sorted by content.
    """
    sessions = record._sessions if record is not None else {}
    participants = ({subject_identifier if named is None else named for named in sessions.values()}
                    if sessions else {subject_identifier})
    stated = record._stated if record is not None else set()
    states = sorted(record._store_states) if record is not None else []
    store = NOT_STATED if not states else states[0] if len(states) == 1 else tuple(states)
    entries = list(record._provenance.values()) if record is not None else []
    entries += [{'subject': participant, 'parts': NOT_STATED, 'activation_store': store}
                for participant in participants - stated]
    return tuple(sorted(entries, key=lambda entry: json.dumps(entry, sort_keys=True, default=str)))


def note_response(event, request=None) -> None:
    """Note a response event a session accepts: a response record, or a perturbation handle.

    ``request`` is the ``StateChange`` the session delivered, if any. The handle
    comes from a typed ``PerturbationApplied`` payload, else the response meta,
    else the request; the response is an undo when the request or the meta says
    ``kind == 'reset'``.
    """
    channel = getattr(event, 'channel', None)
    if is_behavior_channel(channel):
        _note_record(event.payload)
    elif channel == 'perturbation':
        from ..events import PerturbationApplied
        record, meta = _active.get(), event.meta
        if record is None:
            return
        handle = event.payload.handle_id if isinstance(event.payload, PerturbationApplied) else None
        handle = handle or meta.get('handle_id') or getattr(request, 'handle_id', None)
        if handle is not None:
            undo = meta.get('kind') == 'reset' or getattr(request, 'kind', None) == 'reset'
            (record.undone if undo else record.applied)[handle] += 1


def is_behavior_channel(channel) -> bool:
    return isinstance(channel, str) and (channel == 'behavior' or channel.startswith('behavior:'))
