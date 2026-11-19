"""The ``Subject`` base class runs the session duties around every ``participate``.

A minimal subject that only reads and answers gets them: an input on a channel it does
not declare never reaches its body, and a second session on the busy subject is
refused. A nested call with the same session (``super().participate``, a wrapper that
delegates the session) passes.
"""
import threading

import pytest

from brainscore_core.contract import Subject
from brainscore_core.streaming import Session, StreamEvent


class _Reader(Subject):
    """Reads every input and answers once; it calls nothing else."""
    identifier = 'reader'
    in_channels = {'text'}
    out_channels = {'behavior:text'}

    def __init__(self):
        self.seen, self.during = [], None

    def participate(self, session):
        while (event := session.next_input()) is not None:
            self.seen.append(event.channel)
            if self.during is not None:
                self.during()
        session.respond(StreamEvent('behavior:text', 'done', None, {}))


def _session(*channels):
    events = [StreamEvent(channel, 1.0 if channel == 'reward' else 'word', None, {})
              for channel in channels]
    return Session(events, outputs=('behavior:text',))


def test_an_undeclared_input_never_reaches_the_body():
    subject = _Reader()
    subject.participate(_session('reward', 'text', 'reward'))
    assert subject.seen == ['text']


def test_a_second_session_on_the_busy_subject_is_refused_with_a_remedy():
    subject, refusals = _Reader(), []

    def second():
        try:
            subject.participate(_session('text'))
        except RuntimeError as error:
            refusals.append(str(error))

    def during():
        subject.during = None           # only the first session starts another one
        second()

    subject.during = during
    subject.participate(_session('text'))
    assert len(refusals) == 1 and 'reader' in refusals[0] and 'separate subject' in refusals[0]
    subject.participate(_session('text'))       # the subject is free again


def test_a_session_from_another_thread_is_refused():
    subject, refusals = _Reader(), []

    def other_thread():
        try:
            subject.participate(_session('text'))
        except RuntimeError:
            refusals.append(True)

    def during():
        subject.during = None           # only the first session starts another one
        thread = threading.Thread(target=other_thread, daemon=True)
        thread.start()
        thread.join(timeout=10)

    subject.during = during
    subject.participate(_session('text'))
    assert refusals == [True]


def test_a_nested_call_with_the_same_session_passes():
    class Outer(_Reader):
        def participate(self, session):
            super().participate(session)

    class Delegating(Subject):
        identifier = 'delegating'
        in_channels = {'text', 'reward'}
        out_channels = {'behavior:text'}

        def __init__(self, inner):
            self.inner = inner

        def participate(self, session):
            self.inner.participate(session)

    outer = Outer()
    outer.participate(_session('reward', 'text'))
    assert outer.seen == ['text']
    inner = _Reader()
    Delegating(inner).participate(_session('reward', 'text'))
    assert inner.seen == ['text']          # each subject narrows to its own declaration


def test_a_subject_that_still_holds_the_guard_itself_is_told_to_delete_that_line():
    from brainscore_core._sessions.one_at_a_time import one_session_at_a_time

    class HoldsTheGuard(_Reader):
        def participate(self, session):
            with one_session_at_a_time(self):
                super().participate(session)

    with pytest.raises(RuntimeError, match='delete'):
        HoldsTheGuard().participate(_session('text'))

