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
# generator that holds that session. The lock guards only the decision.
_active_sessions = {}
_active_sessions_lock = threading.Lock()
_naming = threading.local()


def one_session_at_a_time(subject):
    """Refuse a second session on ``subject`` while its first one is running.

    The reservation is the guard's own generator: it counts only while that
    generator is running or suspended in the session body. An interrupt at any
    line of the guard ends the generator, so the subject is free at once, also
    when the caller keeps the exception and its traceback. Everything that
    allocates (and so can start a garbage collection that runs a subject's
    finalizer) happens before the lock is taken; the error text, which reads the
    subject's ``identifier``, is built after the lock is released.
    """
    owner = []
    hold = _hold(subject, owner)
    owner.append(weakref.ref(hold))
    return contextmanager(lambda: hold)()


def _hold(subject, owner):
    key, mine = id(subject), owner[0]
    # One line: no line of this frame runs between the claim and the unlock.
    with _active_sessions_lock: running = _claim(key, mine)  # noqa: E701
    busy, running = running is not None, None   # a finalizer this frees runs unlocked
    if busy:
        raise RuntimeError(
            f'Subject {_name(subject)!r} is already taking part in a session. A subject '
            'takes part in one session at a time, because its recording and task state '
            'belong to that session. Wait for the running session to finish, or build a '
            'separate subject for each session that runs at the same time.')
    try:
        yield
    finally:
        # Only this generator replaces its own entry while it runs, so no lock is needed.
        # ponytail: an interrupt inside contextlib's own enter or exit lines keeps the
        # subject busy until the caller drops the exception (the generator then closes).
        if _active_sessions.get(key) is mine:
            del _active_sessions[key]


def _claim(key, mine):
    """The decision, under the lock: no allocation and no subject code.

    Returns the generator of the running session (the caller drops it after the lock),
    or None after it reserves the subject for ``mine``. A holder that is not running is
    closed: its frame is gone, so dropping it here runs no subject code."""
    held = _active_sessions.get(key)
    running = held() if held is not None else None
    if running is not None and (running.gi_running or running.gi_suspended):
        return running
    _active_sessions[key] = mine
    return None


def _name(subject) -> str:
    # A refusal met while the identifier is read (an identifier that starts a session
    # on its own subject) names the type, so refusals nest one level, not without end.
    if getattr(_naming, 'active', False):
        return type(subject).__name__
    _naming.active = True
    try:
        return subject.identifier
    except Exception:       # the identifier of a broken subject must not hide the refusal
        return type(subject).__name__
    finally:
        _naming.active = False
