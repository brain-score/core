"""The shared guard: one session at a time on one subject object."""

import gc
import inspect
import sys
import threading

import pytest

from brainscore_core._sessions import one_at_a_time as module
from brainscore_core.streaming_helpers import one_session_at_a_time

REFUSED = 'already taking part in a session'


class Named:
    def __init__(self, identifier='s'):
        self.identifier = identifier


def busy(subject):
    """True when a new session on ``subject`` is refused now."""
    try:
        with one_session_at_a_time(subject):
            return False
    except RuntimeError as error:
        assert REFUSED in str(error)
        return True


def test_sequential_sessions_run():
    subject = Named()
    for _ in range(3):
        with one_session_at_a_time(subject):
            pass
    assert not busy(subject)


def test_a_nested_session_is_refused_with_the_identifier_and_a_remedy():
    subject = Named('nested')
    with one_session_at_a_time(subject):
        with pytest.raises(RuntimeError, match=REFUSED) as refusal:
            with one_session_at_a_time(subject):
                pass
        assert "'nested'" in str(refusal.value)
        assert 'Wait for the running session to finish' in str(refusal.value)
        assert busy(subject)            # a refusal keeps the running reservation
    assert not busy(subject)


def test_a_session_from_another_thread_is_refused_and_the_first_finishes():
    subject = Named()
    entered, release, outcome = threading.Event(), threading.Event(), {}

    def first():
        with one_session_at_a_time(subject):
            entered.set()
            assert release.wait(3)
        outcome['first'] = 'done'
    thread = threading.Thread(target=first)
    thread.start()
    try:
        assert entered.wait(3)
        assert busy(subject)
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert outcome == {'first': 'done'}
    assert not busy(subject)


def test_different_subjects_run_at_the_same_time():
    a, b = Named('a'), Named('b')
    with one_session_at_a_time(a):
        with one_session_at_a_time(b):
            pass


def test_a_failed_session_releases_the_subject():
    subject = Named()
    with pytest.raises(ValueError):
        with one_session_at_a_time(subject):
            raise ValueError('the session failed')
    assert not busy(subject)


def test_an_interrupt_in_the_session_releases_the_subject():
    subject = Named()
    with pytest.raises(KeyboardInterrupt):
        with one_session_at_a_time(subject):
            raise KeyboardInterrupt
    assert not busy(subject)


def test_the_error_text_is_built_outside_the_lock():
    """An identifier that starts a session on another subject must not deadlock."""
    other = Named('other')

    class Starts:
        @property
        def identifier(self):
            with one_session_at_a_time(other):
                return 'starts'
    subject, outcome = Starts(), {}

    def second():
        try:
            with one_session_at_a_time(subject):
                pass
        except RuntimeError as error:
            outcome['error'] = str(error)
    with one_session_at_a_time(subject):
        thread = threading.Thread(target=second, daemon=True)
        thread.start()
        thread.join(3)
    assert not thread.is_alive(), 'the refusal deadlocked'
    assert "'starts'" in outcome['error']


def _interrupt_at(fragment, subject):
    """Raise KeyboardInterrupt in the guard on the first line after the one with ``fragment``."""
    guard = one_session_at_a_time.__wrapped__
    lines, start = inspect.getsourcelines(guard)
    after = start + next(i for i, line in enumerate(lines) if fragment in line)
    tracer_before = sys.gettrace()      # a coverage tool or a debugger stays active

    def interrupt(frame, event, arg):
        if frame.f_code is guard.__code__ and event == 'line' and frame.f_lineno > after:
            sys.settrace(tracer_before)
            raise KeyboardInterrupt(fragment)
        return interrupt
    try:
        sys.settrace(interrupt)
        with one_session_at_a_time(subject):
            pass
    except KeyboardInterrupt:
        pass
    else:
        pytest.fail('the interrupt did not arrive')
    finally:
        sys.settrace(tracer_before)
    gc.collect()                        # the interrupted call's frame is freed


@pytest.mark.parametrize('fragment', ['_active_sessions[key] =', 'yield'])
def test_an_interrupt_during_reservation_or_release_frees_the_subject(fragment):
    subject = Named()
    _interrupt_at(fragment, subject)
    assert not busy(subject)
    assert id(subject) not in module._active_sessions
