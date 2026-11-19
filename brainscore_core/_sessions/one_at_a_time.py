"""One session at a time on one subject.

A subject's recording, task and change state belong to the one session it takes
part in, so a second session on the same subject object (a nested call, or a call
from another thread) is refused instead of sharing that state. Sessions on
different subject objects run at the same time.
"""

import threading

# id(subject) -> the guard of the session that subject takes part in now. The guard
# holds its subject, so the id cannot go to another object while the entry exists.
_active_sessions = {}
_naming = threading.local()


class one_session_at_a_time:
    """``with one_session_at_a_time(subject):`` refuses a second session on ``subject``
    while its first one runs.

    There is no lock. The claim is ``dict.setdefault`` with an integer key, which is
    one step of the interpreter: no other thread, no garbage collection and no code of
    a subject runs inside it. So a claim cannot wait, and two sessions cannot both
    claim one subject. Only the guard that claimed an entry removes it.

    Limit: an interrupt that arrives inside ``__exit__`` before its first line leaves
    the subject reserved. The refusal's remedy (a separate subject) still works.
    """

    def __init__(self, subject):
        self._subject = subject

    def __enter__(self):
        try:
            claimed = _active_sessions.setdefault(id(self._subject), self) is self
        except BaseException:   # an interrupt right after the claim: give it back
            self._release()
            raise
        if not claimed:
            raise RuntimeError(
                f'Subject {_name(self._subject)} is already taking part in a session. A subject '
                'takes part in one session at a time, because its recording and task state '
                'belong to that session. Wait for the running session to finish, or build a '
                'separate subject for each session that runs at the same time.')

    def __exit__(self, *_):
        self._release()
        return False

    def _release(self):
        if _active_sessions.get(id(self._subject)) is self:
            del _active_sessions[id(self._subject)]


def _name(subject) -> str:
    # The identifier and its text are subject code. If that code starts a session on its
    # own subject, the refusal it meets names the type, so refusals nest one level.
    if getattr(_naming, 'active', False):
        return type(subject).__name__
    _naming.active = True
    try:
        return repr(str(subject.identifier))
    except Exception:       # the identifier of a broken subject must not hide the refusal
        return type(subject).__name__
    finally:
        _naming.active = False
