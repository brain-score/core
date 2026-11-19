"""The shared guard: one session at a time on one subject object."""

import inspect
import os
import subprocess
import sys
import threading
from contextlib import contextmanager

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


@contextmanager
def _trace_guard(on_line):
    """Call ``on_line(line)`` for each line that the guard's generator runs."""
    tracer_before = sys.gettrace()      # a coverage tool or a debugger stays active

    def trace(frame, event, arg):
        code = frame.f_code
        if code.co_filename != module.__file__ or not code.co_flags & inspect.CO_GENERATOR:
            return None

        def local(frame, event, arg):
            if event == 'line':
                on_line(frame.f_lineno)
            return local
        return local
    sys.settrace(trace)
    try:
        yield
    finally:
        sys.settrace(tracer_before)


def _session(subject, on_line):
    """Run one empty session on ``subject``; ``on_line(phase, count)`` sees each guard line,
    ``phase`` being 'enter' before the body and 'exit' after it."""
    phase, count = ['enter'], {'enter': 0, 'exit': 0}

    def line_run(line):
        count[phase[0]] += 1
        on_line(phase[0], count[phase[0]])
    with _trace_guard(line_run):
        with one_session_at_a_time(subject):
            phase[0] = 'exit'
    return count


def _runs_soon(subject):
    """True when the next session on ``subject`` runs within seconds (a held lock fails)."""
    outcome = []
    thread = threading.Thread(target=lambda: outcome.append(busy(subject)), daemon=True)
    thread.start()
    thread.join(5)
    return outcome == [False]


def test_an_interrupt_at_any_line_of_the_guard_keeps_the_subject_free():
    """Both boundaries included: after the reservation, and after the body before the
    release. The caller keeps each interrupt (so its traceback); the next session runs."""
    lines = _session(Named(), lambda phase, count: None)
    assert lines['enter'] and lines['exit']
    kept = []
    for phase, total in lines.items():
        for at in range(1, total + 1):
            subject = Named()

            def interrupt(now, count):
                if (now, count) == (phase, at):
                    raise KeyboardInterrupt(f'{phase} line {at}')
            with pytest.raises(KeyboardInterrupt) as info:
                _session(subject, interrupt)
            kept.append(info.value)
            assert _runs_soon(subject), f'no session after an interrupt at {phase} line {at}'
    assert all(error.__traceback__ is not None for error in kept)


FINALIZER_STARTS_A_SESSION = """
import gc
from brainscore_core.streaming_helpers import one_session_at_a_time

class Named:
    def __init__(self, identifier='s'):
        self.identifier = identifier

class StartsWhenCollected:
    def __init__(self):
        self.cycle = self

    def __del__(self):
        with one_session_at_a_time(Named('collected')):
            pass

def add_garbage(phase, info):
    if phase == 'stop':                 # two new objects: the next allocation collects again
        StartsWhenCollected()
        StartsWhenCollected()

gc.callbacks.append(add_garbage)
gc.set_threshold(1)                     # so each allocation runs a collection
for _ in range(20):
    with one_session_at_a_time(Named()):
        pass
gc.set_threshold(700)
print('done')
"""


def test_a_finalizer_that_starts_a_session_does_not_deadlock():
    """A collection while a session starts runs a subject's finalizer, which starts a session.

    A fresh process, so that a deadlock fails this test without holding the guard here."""
    try:
        run = subprocess.run([sys.executable, '-c', FINALIZER_STARTS_A_SESSION],
                             capture_output=True, text=True, timeout=20,
                             env={**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path)})
    except subprocess.TimeoutExpired:
        pytest.fail('a finalizer deadlocked on the guard')
    assert run.returncode == 0, run.stderr
    assert 'done' in run.stdout
