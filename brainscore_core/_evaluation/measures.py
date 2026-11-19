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
where a session accepts its first response (``note_session``): the delivery form
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

A session's notes are staged: the ``Subject`` base runs each participation, the claim of
the subject and its release included, inside its own nested record (``stage_session``), so
the notes reach the score only when the outermost participation of the session returns
without an error, all at once. An intervention response is counted at once in every
open record, with the subject that gave it: a failed session cannot hide a change it left
active, and an undo cancels only a change of the same subject.

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
from contextvars import ContextVar, copy_context
from typing import Any, Callable, Dict, Iterator, Optional, Set, Tuple

_active: ContextVar[Optional['MeasureRecord']] = ContextVar('measure_record', default=None)


class _Facts:
    """The descriptive facts of a record (everything but the intervention counts)."""

    def __init__(self) -> None:
        self.records: Dict[str, dict] = {}       # one per distinct record, keyed by its JSON
        self.deliveries: Set[str] = set()
        self.region_plans: Set[str] = set()
        self.provenance: Dict[str, dict] = {}    # one per distinct session entry, keyed by its JSON
        self.store_states: Set[str] = set()
        self.sessions: Dict[int, Optional[str]] = {}   # session key -> the subject that took part, if named
        self.stated: Set[str] = set()            # subjects that stated their parts
        self.answered: Set[int] = set()          # sessions whose first response is noted

    def add(self, other: '_Facts') -> None:
        self.records.update(other.records)
        self.deliveries |= other.deliveries
        self.region_plans |= other.region_plans
        self.provenance.update(other.provenance)
        self.store_states |= other.store_states
        for key, named in other.sessions.items():
            if named is not None or key not in self.sessions:
                self.sessions[key] = named
        self.stated |= other.stated
        self.answered |= other.answered


class MeasureRecord:
    """The response records seen during one scoring run, and how often its
    sessions applied and undid each perturbation handle (``perturbation``
    responses carry a ``handle_id``; ``meta['kind'] == 'reset'`` marks an undo).
    Counts, not sets: a handle name used again after its undo is counted again.

    A record changes only by single steps that an interrupt cannot split: a note adds
    one entry; a nested record that succeeded publishes all its facts with one append
    (``_published``); an intervention response is counted at once in this record and
    in every record around it, with the subject that gave it, so no merge can lose it
    and an undo cancels only a change of the same subject.
    """

    def __init__(self, outer: Optional['MeasureRecord'] = None, participant=None) -> None:
        self._outer = outer
        self._participant = participant      # the subject of a staged participation
        self._facts = _Facts()               # what was noted while this record was the open one
        self._published: list = []           # the facts of nested records that succeeded
        self._changes: list = []             # (subject, handle, undo) of each intervention response

    def _all_facts(self) -> _Facts:
        facts = _Facts()
        facts.add(self._facts)
        for published in list(self._published):
            facts.add(published)
        return facts

    def _counts(self, undo: bool, by_subject: bool = False) -> Counter:
        return Counter((id(subject), handle) if by_subject else handle
                       for subject, handle, is_undo in list(self._changes) if is_undo == undo)

    @property
    def applied(self) -> Counter:
        """How often the run's sessions applied each handle."""
        return self._counts(undo=False)

    @property
    def undone(self) -> Counter:
        """How often the run's sessions undid each handle."""
        return self._counts(undo=True)

    @property
    def unreleased(self) -> Set[str]:
        """Handles applied during the run more often than the same subject undid them."""
        return {handle for _, handle in self._counts(False, True) - self._counts(True, True)}

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
        records = self._all_facts().records
        return tuple(records[key] for key in sorted(records, key=lambda key: (records[key]['format'], key)))

    @property
    def provenance(self) -> Tuple[dict, ...]:
        """The distinct provenance entries the run's sessions noted, sorted by content."""
        entries = self._all_facts().provenance
        return tuple(entries[key] for key in sorted(entries))


@contextmanager
def record_measures(subject: Optional[str] = None) -> Iterator[MeasureRecord]:
    """Collect the measure of each behavioral response until the block ends.

    A record opened inside another one (a benchmark that calls ``score()``, or
    the record of one session) passes its measures to the outer record when it
    ends without an error, so the outer score describes the responses its result
    came from. A failed inner run passes on only its intervention counts: its
    result does not reach the outer score, but a change it left active does
    reach the release check.

    ``subject`` is the identifier of the subject the score runs. When the block
    ends, every session no note named is bound to it, before the record passes
    to an outer one: a session belongs to the subject of the score that ran it.
    """
    outer = _active.get()
    record = MeasureRecord(outer)
    token = _active.set(record)
    try:
        yield record
    finally:
        # ponytail: one step is not protected: an interrupt as this block starts leaves
        # this record open in the caller's context. Later notes there then reach only this
        # record (lost to the outer score, never published from a failed run); counts still
        # reach every enclosing record. Python has no context manager in C that resets a
        # ContextVar; a participation avoids the step (stage_session runs in a copied context).
        try:
            _active.reset(token)
        finally:
            _active.set(outer)
    _publish(record, outer, subject)


def stage_session(session, participant, run: Callable[[], Any]) -> Any:
    """Return ``run()`` with what ``session`` notes meanwhile staged; name it on success.

    The ``Subject`` base runs every participation that claims a subject through this
    call, the claim and its release included. The notes go to a nested record that is
    open only in a copy of the context, so the outer record is open again however
    ``run`` ends. When ``run`` returns, the session is named after ``participant`` (its
    ``identifier``), last, and the notes reach the outer record at once: the outermost
    subject that was given the session names it, so a wrapper names its sessions, not
    the subject inside it. When ``run`` raises, only the intervention counts (made at
    once in every record) remain. Without an open record (a session outside a score)
    nothing is kept.
    """
    outer = _active.get()
    if outer is None:
        return run()
    record = MeasureRecord(outer, participant)
    result = copy_context().run(_open_and_run, record, run)
    _publish(record, outer, participant.identifier, session)
    return result


def _open_and_run(record: MeasureRecord, run: Callable[[], Any]) -> Any:
    _active.set(record)
    return run()


def _publish(record: MeasureRecord, outer: Optional[MeasureRecord], name: Optional[str],
             session=None) -> None:
    """Pass the facts of ``record``, which succeeded, to ``outer`` in one append, after
    ``name`` named ``session`` (or, without a session, every session no note named). An
    interrupt before that append leaves ``outer`` as it was."""
    facts = record._all_facts()
    if name is not None:
        keys = [_session_key(session)] if session is not None else [
            key for key, named in facts.sessions.items() if named is None]
        facts.sessions.update(dict.fromkeys(keys, name))
    if outer is None:
        record._facts, record._published = facts, []
    else:
        outer._published.append(facts)


def _note_record(payload) -> None:
    """Add ``payload.attrs['response']`` to the open record, if any."""
    record = _active.get()
    if record is None:
        return
    if isinstance(payload, dict):                  # a stored cell: its checked row
        response = payload.get('response')
    else:
        attrs = getattr(payload, 'attrs', None)
        response = attrs.get('response') if isinstance(attrs, dict) else None
    if isinstance(response, dict) and 'format' in response:
        record._facts.records[json.dumps(response, sort_keys=True, default=str)] = response


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
                 provenance: Optional[dict] = None) -> None:
    """Note how ``session`` delivers its inputs and, when given, the plan the subject used
    and the provenance of the session.

    ``regions_per_pass`` comes from the subject that owns the setting and validated
    it (``BrainScoreModel``); a subject that does not read it passes none.
    ``provenance`` is one entry (a mapping of plain values with a ``subject``) from
    the subject that owns the parts; core adds ``activation_store`` (the store state
    of the session) and keeps a copy. An entry with ``parts`` says that its subject
    stated its parts. The subject that takes part in ``session`` names it when its
    participation succeeds (``stage_session``); a session that no subject names belongs
    to the scored subject. Each subject that took part and
    stated no parts gets an entry with its facts ``'not stated'`` (``provenance_attrs``).
    """
    record = _active.get()
    if record is None:
        return
    record._facts.deliveries.add('stream' if session.streaming else 'listed')
    record._facts.store_states.add(_store_state())
    if regions_per_pass is not None:
        record._facts.region_plans.add(regions_per_pass)
    key = _session_key(session)
    named = (provenance or {}).get('subject')
    if named is not None or key not in record._facts.sessions:
        record._facts.sessions[key] = named if named is not None else record._facts.sessions.get(key)
    if provenance is not None:
        entry = {**deepcopy(dict(provenance)), 'activation_store': _store_state()}
        record._facts.provenance[json.dumps(entry, sort_keys=True, default=str)] = entry
        if 'parts' in entry:
            record._facts.stated.add(entry.get('subject'))


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
    facts = record._all_facts()
    return {'delivery': one(facts.deliveries, None),
            'regions_per_pass': one(facts.region_plans, None)}


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
    facts = record._all_facts() if record is not None else _Facts()
    sessions = facts.sessions
    participants = ({subject_identifier if named is None else named for named in sessions.values()}
                    if sessions else {subject_identifier})
    stated = facts.stated
    states = sorted(facts.store_states)
    store = NOT_STATED if not states else states[0] if len(states) == 1 else tuple(states)
    entries = list(facts.provenance.values())
    entries += [{'subject': participant, 'parts': NOT_STATED, 'activation_store': store}
                for participant in participants - stated]
    return tuple(sorted(entries, key=lambda entry: json.dumps(entry, sort_keys=True, default=str)))


def note_response(event, request=None, session=None) -> None:
    """Note a response event a session accepts: a response record, or a perturbation handle.

    The first response of ``session`` that the open record sees also notes the session
    (``note_session``): once for each session in each record, so a failed attempt that
    was dropped does not hide the delivery of a later good one.

    ``request`` is the ``StateChange`` the session delivered, if any. The handle
    comes from a typed ``PerturbationApplied`` payload, else the response meta,
    else the request; the response is an undo when the request or the meta says
    ``kind == 'reset'``.
    """
    record = _active.get()
    if session is not None and record is not None and _session_key(session) not in record._facts.answered:
        record._facts.answered.add(_session_key(session))
        note_session(session)
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
            _count(record, handle, undo)


def _count(record: MeasureRecord, handle, undo: bool) -> None:
    """Count one intervention response in ``record`` and in every record around it, the
    outermost first, with the subject that gave it: the subject of the innermost staged
    participation (None for a session that no ``Subject`` runs)."""
    chain = []
    while record is not None:
        chain.append(record)
        record = record._outer
    subject = next((each._participant for each in chain if each._participant is not None), None)
    for each in reversed(chain):
        each._changes.append((subject, handle, undo))


def is_behavior_channel(channel) -> bool:
    return isinstance(channel, str) and (channel == 'behavior' or channel.startswith('behavior:'))
