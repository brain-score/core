"""One session at a time on one subject (``brainscore_core.streaming_helpers.one_session_at_a_time``)."""
import gc
import threading

import pytest

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
    assert most[0] == 1 and _runs(subject)


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

    thread = threading.Thread(target=session)
    thread.start()
    thread.join(timeout=20)
    assert done, 'a session waited on a finalizer'


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
