"""Record how the behavioral responses of one scoring run were measured.

A behavioral assembly states its measure in ``attrs['measure']``:
``{'kind': 'reported' | 'implied', 'source': ...}`` (see
``brainscore_core.responses``). ``score()`` opens ``record_measures()`` around
the benchmark run. Core notes the measure of each behavioral response it hands
to a benchmark: the legacy facade, the typed helpers and
``BrainScoreModel.process``. The score metadata then reads the record. Only
returned data is read, never the subject's state.
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
    """Collect the measure of each behavioral response until the block ends."""
    record = MeasureRecord()
    token = _active.set(record)
    try:
        yield record
    finally:
        _active.reset(token)


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


def is_behavior_channel(channel: str) -> bool:
    return channel == 'behavior' or str(channel).startswith('behavior:')

