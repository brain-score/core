"""One session at a time on one subject.

A subject's recording, task and change state belong to the one session it takes
part in, so a second session on the same subject object (a nested call, or a call
from another thread) is refused instead of sharing that state. Sessions on
different subject objects run at the same time.
"""

import threading
import weakref
from contextlib import contextmanager

# Subjects (by id) that take part in a session now -> a weak reference to the
# reservation of that session. The lock guards only the check and the reservation.
_active_sessions = {}
_active_sessions_lock = threading.Lock()


class _Reservation:
    """The running session's hold on its subject. While it lives, the subject is busy."""


@contextmanager
def one_session_at_a_time(subject):
    """Refuse a second session on ``subject`` while its first one is running.

    The decision is made under a lock; the error text (which reads the subject's
    ``identifier``) is built after the lock is released. The release needs no lock.
    A table entry counts only while its reservation object lives, so an interrupt
    that arrives during the reservation or the release frees the subject when the
    interrupted call's frame is freed. A refused call never releases the running one.
    """
    key, reservation = id(subject), _Reservation()
    with _active_sessions_lock:
        held = _active_sessions.get(key)
        busy = held is not None and held() is not None
        if not busy:
            _active_sessions[key] = weakref.ref(reservation)
    if busy:
        raise RuntimeError(
            f'Subject {_name(subject)!r} is already taking part in a session. A subject '
            'takes part in one session at a time, because its recording and task state '
            'belong to that session. Wait for the running session to finish, or build a '
            'separate subject for each session that runs at the same time.')
    try:
        yield
    finally:
        # ponytail: one atomic dict call, no lock; a stranded entry dies with `reservation`.
        _active_sessions.pop(key, None)


def _name(subject) -> str:
    try:
        return subject.identifier
    except Exception:       # the identifier of a broken subject must not hide the refusal
        return type(subject).__name__
