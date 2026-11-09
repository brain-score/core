"""Record how the behavioral responses of one scoring run were measured.

A behavioral assembly states its measure in ``attrs['measure']``:
``{'kind': 'reported' | 'implied', 'source': ...}`` (see
``brainscore_core.responses``). ``score()`` opens ``record_measures()`` around
the benchmark run. Core notes the measure where a session accepts a response
from the subject (``note_response`` in each core session's ``respond`` /
``emit``), event by event, before any session concatenates payloads. Events
that no session accepts (saved records, fixtures, copies) are not noted.
``BrainScoreModel.process`` notes the assembly it returns, because that typed
path has no session. The score metadata then reads the record. Only returned
data is read, never the subject's state.

ponytail: the record is a ``ContextVar``. A response made in a worker thread or
process is not noted. No behavioral path in core, vision, language or unified
makes its response events there today (worker pools only extract features or
compute metrics); copy the context into the worker if one ever does.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional, Set, Tuple

MEASURE_KINDS = ('reported', 'implied')

_active: ContextVar[Optional['MeasureRecord']] = ContextVar('measure_record', default=None)


class MeasureRecord:
    """The ``(kind, source)`` pairs seen during one scoring run."""

    def __init__(self) -> None:
        self.measures: Set[Tuple[str, Optional[str]]] = set()

    @property
    def kinds(self) -> Set[str]:
        return {kind for kind, _source in self.measures}

    @property
    def sources(self) -> Set[Optional[str]]:
        return {source for _kind, source in self.measures}


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
            outer.measures |= record.measures


def note_measure(assembly) -> None:
    """Add ``assembly.attrs['measure']`` to the open record, if any."""
    record = _active.get()
    if record is None:
        return
    attrs = getattr(assembly, 'attrs', None)
    measure = attrs.get('measure') if isinstance(attrs, dict) else None
    if isinstance(measure, dict) and measure.get('kind') in MEASURE_KINDS:
        source = measure.get('source')
        record.measures.add((measure['kind'], None if source is None else str(source)))


def note_response(event) -> None:
    """Note a response event a session accepts, if it is on a behavior channel."""
    if is_behavior_channel(getattr(event, 'channel', None)):
        note_measure(event.payload)


def is_behavior_channel(channel) -> bool:
    return isinstance(channel, str) and (channel == 'behavior' or channel.startswith('behavior:'))
