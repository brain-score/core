"""One session at a time on one subject.

The ``Subject`` base class claims this guard around every ``participate``
(``brainscore_core.contract``); a subject does not use it itself.

A subject's recording, task and change state belong to the one session it takes
part in, so a second session on the same subject object (a nested call, or a call
from another thread) is refused instead of sharing that state. Sessions on
different subject objects run at the same time.
"""

import threading
from contextlib import contextmanager
from types import TracebackType
from typing import TYPE_CHECKING, Optional, Type

if TYPE_CHECKING:
    from brainscore_core.contract import Subject

# id(subject) -> the token (subject, thread id) of the session that subject takes part in
# now. The token holds its subject, so the id cannot go to another object while the entry exists.
_active_sessions = {}
_naming = threading.local()


class one_session_at_a_time:
    """``with one_session_at_a_time(subject):`` refuses a second session on ``subject``
    while its first one runs. Make a new guard for each session: a guard is used once.

    There is no lock. The claim is ``dict.setdefault`` with an integer key, which is
    one step of the interpreter: no other thread, no garbage collection and no code of
    a subject runs inside it. So a claim cannot wait, and two sessions cannot both
    claim one subject. Only the session that claimed an entry removes it.

    Interrupts (one interrupt, CPython 3.11): in the protected claim, in the session body
    and in the protected release the subject is free at once. If the session's generator
    stays suspended after the enter returns or before the exit resumes it, the subject is
    free when the guard and the interrupted call's traceback lose their last references.
    Limit: an interrupt at the generator's own yield or resume event, or after the body
    ends and before the protected release, leaves the subject reserved for the life of the
    process (no step can safely undo a claim for another caller). The reserved entry holds
    that subject, so no other object is refused in its place: a separate subject runs.
    """

    def __init__(self, subject: 'Subject') -> None:
        self._session = _session(subject)
        self._used = False

    def __enter__(self) -> None:
        if self._used:      # a second entry would end the first session's claim
            raise RuntimeError('This guard is already in use. Make a new one_session_at_a_time '
                               'guard for each session.')
        self._used = True
        return self._session.__enter__()

    def __exit__(self, error_type: Optional[Type[BaseException]], error: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> bool:
        return self._session.__exit__(error_type, error, traceback)


@contextmanager
def _session(subject):
    key, token = id(subject), (subject, threading.get_ident())
    try:    # the claim and the session body are in one try, so every way out releases
        if (held := _active_sessions.setdefault(key, token)) is not token:
            raise RuntimeError(
                f'Subject {_name(subject)} is already taking part in a session. A subject '
                'takes part in one session at a time, because its recording and task state '
                'belong to that session. Wait for the running session to finish, or build a '
                'separate subject for each session that runs at the same time.'
                + (_SAME_THREAD if held[1] == token[1] else ''))
        yield
    finally:
        try:
            _release(key, token)
        except BaseException:   # an interrupt inside the release: finish it, then pass it on
            _release(key, token)
            raise


_SAME_THREAD = (' The running session is on this thread: if the participate of this subject '
                'claims one_session_at_a_time itself, delete that line; the Subject base class '
                'claims the session around participate.')


def _release(key, token):
    if _active_sessions.get(key) is token:
        del _active_sessions[key]


def _type_name(subject) -> str:
    return type.__dict__['__name__'].__get__(type(subject))   # not through a metaclass


def _name(subject) -> str:
    # The identifier and its text are subject code. If that code starts a session on its
    # own subject, the refusal it meets names the type, so refusals nest one level.
    if getattr(_naming, 'active', False):
        return _type_name(subject)
    _naming.active = True
    try:
        return repr(str(subject.identifier))
    except Exception:       # the identifier of a broken subject must not hide the refusal
        return _type_name(subject)
    finally:
        _naming.active = False
