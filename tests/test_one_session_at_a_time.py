"""One session at a time on one subject (``brainscore_core.streaming_helpers.one_session_at_a_time``)."""
import gc
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


def _interrupted_session(subject, at):
    """Run one session and raise an interrupt at the ``at``-th line that the guard's own code
    runs. Returns (fired, in the stated limit, the kept exception)."""
    seen, state, kept = [0], {'body': False, 'release': False, 'limit': False, 'fired': False}, None

    def trace(frame, event, argument):
        if frame.f_code.co_filename != helper.__file__:
            return None
        if frame.f_code.co_name == '_release':
            state['release'] = True
        if event == 'line' and not state['fired']:
            seen[0] += 1
            if seen[0] == at:
                state['fired'] = True
                # The stated limit: after the body ended and before the release started.
                state['limit'] = state['body'] and not state['release']
                raise KeyboardInterrupt
        return trace

    sys.settrace(trace)
    try:
        with one_session_at_a_time(subject):
            state['body'] = True
    except KeyboardInterrupt as error:
        kept = error
    finally:
        sys.settrace(None)
    return state['fired'], state['limit'], kept


def test_an_interrupt_in_the_guard_frees_the_subject_outside_the_stated_limit():
    fired_count = 0
    for at in range(1, 200):
        subject = _Subject()
        fired, in_limit, kept = _interrupted_session(subject, at)
        if not fired:
            break
        fired_count += 1
        del kept
        gc.collect()        # the interrupted call's exception and guard are gone
        if in_limit:
            assert _runs(_Subject()), 'a separate subject must run'
        else:
            assert _runs(subject), f'the subject stays reserved after an interrupt at guard line {at}'
    assert fired_count >= 5, 'the interrupts did not reach the guard'


def test_a_subject_left_reserved_by_an_interrupt_does_not_block_a_new_object():
    # The entry holds its subject, so a new object cannot get the reserved id.
    stranded = [at for at in range(1, 200) if _interrupted_session(_Subject(), at)[1]]
    gc.collect()
    assert all(_runs(_Subject()) for _ in range(50)) or not stranded


_FINALIZERS_DURING_ENTRY = """
import gc, sys
from brainscore_core.streaming_helpers import one_session_at_a_time

class Subject:
    identifier = 'stand-in'

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
