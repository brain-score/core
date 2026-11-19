"""One session at a time on one subject (``brainscore_core.streaming_helpers.one_session_at_a_time``)."""
import gc
import inspect
import sys
import threading

import pytest

from brainscore_core._sessions import one_at_a_time as helper
from brainscore_core.streaming_helpers import one_session_at_a_time


class _Subject:
    identifier = 'stand-in'


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
        thread = threading.Thread(target=second)
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

    threads = [threading.Thread(target=session) for _ in range(8)]
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


def _guard_lines():
    """The lines where an interrupt must free the subject: the enter, the claim and the session
    body, and the release. Not the exit before the release starts (the stated limit)."""
    for function, until in ((helper.one_session_at_a_time.__enter__, None),
                            (helper._session.__wrapped__, 'finally:'), (helper._release, None)):
        lines, first = inspect.getsourcelines(function)
        for offset, text in enumerate(lines):
            if until and text.strip() == until:
                break
            if offset:
                yield function.__code__, first + offset


@pytest.mark.parametrize('code,line', list(_guard_lines()))
def test_an_interrupt_at_a_line_of_the_guard_frees_the_subject_when_its_exception_is_dropped(code, line):
    subject, fired = _Subject(), []

    def trace(frame, event, argument):
        if frame.f_code is not code:
            return None
        if event == 'line' and frame.f_lineno == line and not fired:
            fired.append(True)
            raise KeyboardInterrupt
        return trace

    sys.settrace(trace)
    try:
        with one_session_at_a_time(subject):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        sys.settrace(None)
    gc.collect()    # the interrupted call's exception and guard are gone here
    assert _runs(subject)


def test_different_subjects_run_at_the_same_time():
    with one_session_at_a_time(_Subject()):
        assert _runs(_Subject())


@pytest.mark.parametrize('error', [ValueError, KeyboardInterrupt])
def test_a_session_that_ends_with_an_error_or_an_interrupt_releases_the_subject(error):
    subject = _Subject()
    with pytest.raises(error) as kept:   # the exception stays alive during the retry
        with one_session_at_a_time(subject):
            raise error()
    assert _runs(subject) and kept.value is not None


def test_sessions_start_while_automatic_collections_run_finalizers_that_start_sessions():
    class Cycle:
        def __init__(self, subject):
            self.subject, self.me = subject, self

        def __del__(self):
            _runs(self.subject)

    other, done, threshold = _Subject(), [], gc.get_threshold()

    def session():
        for _ in range(300):
            Cycle(other)                    # garbage whose finalizer starts a session
            assert _runs(_Subject())        # a collection can start inside this entry
        done.append(True)

    gc.set_threshold(1)
    try:
        thread = threading.Thread(target=session, daemon=True)
        thread.start()
        thread.join(timeout=30)
    finally:
        gc.set_threshold(*threshold)
    assert done and not thread.is_alive(), 'a session waited on a finalizer'


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

    class Reentrant:
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

    class WithText:
        identifier = Text('text')

    subject = WithText()
    with one_session_at_a_time(subject):
        with pytest.raises(RuntimeError):
            _runs(subject)
    assert len(reads) == 1


def test_a_type_whose_metaclass_hides_its_name_still_gets_the_refusal():
    class Meta(type):
        @property
        def __name__(cls):
            raise LookupError('no name')

    class Odd(metaclass=Meta):
        @property
        def identifier(self):
            raise ValueError('broken')

    subject = Odd()
    with one_session_at_a_time(subject):
        with pytest.raises(RuntimeError):
            _runs(subject)
