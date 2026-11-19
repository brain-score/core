"""One session at a time on one subject (core-private ``_sessions.one_at_a_time``; the ``Subject`` base claims it)."""
import gc
import sys
import threading
import weakref

import pytest

from brainscore_core.contract import Subject
from brainscore_core._sessions.one_at_a_time import one_session_at_a_time


class _Subject(Subject):
    identifier = 'stand-in'
    in_channels = {'text'}
    out_channels = {'behavior:choice'}

    def participate(self, session) -> None:
        pass


def _runs(subject) -> bool:
    """True when a session on ``subject`` can start now."""
    with one_session_at_a_time(subject):
        return True


def test_sequential_sessions_run():
    subject = _Subject()
    assert _runs(subject) and _runs(subject)


def test_a_nested_session_is_refused_with_the_identifier_and_a_remedy():
    subject = _Subject()
    with one_session_at_a_time(subject):
        with pytest.raises(RuntimeError) as refusal:
            _runs(subject)
    assert 'stand-in' in str(refusal.value) and 'separate subject' in str(refusal.value)
    assert _runs(subject)


def test_a_session_from_another_thread_is_refused_and_the_first_finishes():
    subject, refused = _Subject(), []

    def second():
        try:
            _runs(subject)
        except RuntimeError:
            refused.append(True)

    with one_session_at_a_time(subject):
        thread = threading.Thread(target=second, daemon=True)
        thread.start()
        thread.join(timeout=10)
    assert refused and not thread.is_alive()
    assert _runs(subject)


def test_many_threads_claim_one_subject_and_exactly_one_runs_at_a_time():
    subject, inside, most, start = _Subject(), [0], [0], threading.Barrier(8)

    def session():
        start.wait(timeout=10)
        for _ in range(200):
            try:
                with one_session_at_a_time(subject):
                    inside[0] += 1
                    most[0] = max(most[0], inside[0])
                    inside[0] -= 1
            except RuntimeError:
                pass

    threads = [threading.Thread(target=session, daemon=True) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads)
    assert most[0] == 1 and _runs(subject)


def test_a_guard_that_is_entered_a_second_time_is_refused_and_keeps_the_first_claim():
    subject = _Subject()
    guard = one_session_at_a_time(subject)
    with guard:
        with pytest.raises(RuntimeError):
            with guard:
                pass
        with pytest.raises(RuntimeError):   # the first session still holds the subject
            _runs(subject)
    assert _runs(subject)


def test_three_callers_in_a_forced_order_never_share_the_subject():
    subject, a_inside, a_may_leave, order = _Subject(), threading.Event(), threading.Event(), []

    def caller_a():
        with one_session_at_a_time(subject):
            a_inside.set()
            a_may_leave.wait(timeout=10)
        order.append('a left')

    thread = threading.Thread(target=caller_a, daemon=True)
    thread.start()
    assert a_inside.wait(timeout=10)
    with pytest.raises(RuntimeError):       # B, while A is inside
        _runs(subject)
    a_may_leave.set()
    thread.join(timeout=10)
    assert order and not thread.is_alive()
    with one_session_at_a_time(subject):    # B again, after A left
        with pytest.raises(RuntimeError):   # C, while B is inside
            _runs(subject)
    assert _runs(subject)


def test_an_interrupt_in_the_session_body_comes_out_and_frees_the_subject_at_once():
    subject = _Subject()
    with pytest.raises(KeyboardInterrupt) as interrupt:
        with one_session_at_a_time(subject):
            raise KeyboardInterrupt
    assert interrupt.value is not None and _runs(subject)      # the exception is still held


def test_a_session_that_never_reaches_its_exit_frees_the_subject_when_its_guard_is_gone():
    subject = _Subject()
    guard = one_session_at_a_time(subject)
    guard.__enter__()           # as after an interrupt between the enter and the body
    with pytest.raises(RuntimeError):
        _runs(subject)
    del guard
    gc.collect()
    assert _runs(subject)


def _interrupted_session(subject, at):
    """Run one session and raise an interrupt at the ``at``-th event (call, line or return)
    of the guard's own code. Returns (the event kind, True when the body had ended, the
    exception that came out), or None when the session has fewer events."""
    guard_file = sys.modules[one_session_at_a_time.__module__].__file__
    seen, body_ended, fired, kept = [0], [False], [], None

    def trace(frame, event, argument):
        if frame.f_code.co_filename != guard_file:
            return None
        if not fired and event in ('call', 'line', 'return'):
            seen[0] += 1
            if seen[0] == at:
                fired.append((event, body_ended[0]))
                raise KeyboardInterrupt
        return trace

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        with one_session_at_a_time(subject):
            body_ended[0] = True
    except KeyboardInterrupt as error:
        kept = error
    finally:
        sys.settrace(previous)
    return (*fired[0], kept) if fired else None


def test_an_interrupt_at_each_event_of_the_guard_has_one_of_the_three_stated_outcomes():
    freed_at_once_after_the_body, freed_when_the_call_is_gone, reserved = [], [], []
    for at in range(1, 400):
        subject = _Subject()
        interrupted = _interrupted_session(subject, at)
        if interrupted is None:
            break
        event, body_ended, kept = interrupted
        assert isinstance(kept, KeyboardInterrupt), f'the interrupt at event {at} did not come out'
        if _runs_or_refused(subject):               # the exception is still held
            if body_ended and event == 'line':      # a line of the release was interrupted
                freed_at_once_after_the_body.append(at)
            continue
        del kept, interrupted
        gc.collect()                                # the interrupted call is gone
        if _runs_or_refused(subject):
            freed_when_the_call_is_gone.append(at)
            continue
        # The stated limit: the subject stays reserved, and the reservation holds it.
        assert body_ended or event != 'line', f'a subject stays reserved after an interrupt at line event {at}'
        still_there = weakref.ref(subject)
        del subject
        gc.collect()
        assert still_there() is not None, 'a reserved subject was freed: another object could be refused in its place'
        assert not _runs_or_refused(still_there())
        assert all(_runs(_Subject()) for _ in range(20)), 'a separate subject must run'
        reserved.append(at)
    assert freed_at_once_after_the_body, 'no interrupt reached the protected release'
    assert freed_when_the_call_is_gone, 'no interrupt left a session suspended'
    assert reserved, 'no interrupt reached the stated limit'


def _runs_or_refused(subject) -> bool:
    try:
        return _runs(subject)
    except RuntimeError:
        return False


_FINALIZERS_DURING_ENTRY = """
import gc, sys
from brainscore_core.contract import Subject as ContractSubject
from brainscore_core._sessions.one_at_a_time import one_session_at_a_time

class Subject(ContractSubject):
    identifier = 'stand-in'
    in_channels = {'text'}
    out_channels = {'behavior:choice'}

    def participate(self, session) -> None:
        pass

class Cycle:
    def __init__(self, subject):
        self.subject, self.me = subject, self
    def __del__(self):
        with one_session_at_a_time(self.subject):
            pass

other = Subject()
gc.set_threshold(1)
sys.settrace(lambda frame, event, argument: None)   # allocations, so collections, at every call
for _ in range(300):
    Cycle(other)                    # garbage whose finalizer starts a session
    with one_session_at_a_time(Subject()):
        pass
print('done')
"""


def test_sessions_start_while_automatic_collections_run_finalizers_that_start_sessions():
    # In a child process with a hard timeout: a guard that waits here cannot block the suite.
    import os
    import subprocess
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run([sys.executable, '-c', _FINALIZERS_DURING_ENTRY], env=environment,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0 and 'done' in result.stdout, result.stderr[-400:]


def test_a_finalizer_that_starts_a_session_while_a_session_starts_does_not_wait():
    class Cycle:
        def __init__(self, subject):
            self.subject, self.me = subject, self

        def __del__(self):
            _runs(self.subject)

    other, done = _Subject(), []

    def session():
        for _ in range(50):
            Cycle(other)
            gc.collect()        # the finalizer starts a session in the middle of this thread's work
            assert _runs(_Subject())
        done.append(True)

    thread = threading.Thread(target=session, daemon=True)
    thread.start()
    thread.join(timeout=20)
    assert done and not thread.is_alive(), 'a session waited on a finalizer'


def test_an_identifier_that_starts_a_session_on_its_own_subject_is_read_once_for_a_refusal():
    reads = []

    class Reentrant(_Subject):
        @property
        def identifier(self):
            reads.append(True)
            _runs(self)
            return 'unreachable'

    subject = Reentrant()
    with one_session_at_a_time(subject):
        with pytest.raises(RuntimeError) as refusal:
            _runs(subject)
    assert len(reads) == 1 and 'Reentrant' in str(refusal.value)


def test_an_identifier_text_that_starts_a_session_is_formatted_once():
    reads = []

    class Text(str):
        def __str__(self):
            reads.append(True)
            _runs(subject)
            return 'unreachable'

    class WithText(_Subject):
        identifier = Text('text')

    subject = WithText()
    with one_session_at_a_time(subject):
        with pytest.raises(RuntimeError):
            _runs(subject)
    assert len(reads) == 1


def test_a_type_whose_metaclass_hides_its_name_still_gets_the_refusal():
    class Meta(type(_Subject)):
        @property
        def __name__(cls):
            raise LookupError('no name')

    class Odd(_Subject, metaclass=Meta):
        @property
        def identifier(self):
            raise ValueError('broken')

    subject = Odd()
    with one_session_at_a_time(subject):
        with pytest.raises(RuntimeError):
            _runs(subject)
