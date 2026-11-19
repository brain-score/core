"""One session at a time on one subject.

A subject's recording, task and change state belong to the one session it takes
part in, so a second session on the same subject object (a nested call, or a call
from another thread) is refused instead of sharing that state. Sessions on
different subject objects run at the same time.
"""

import threading
from contextlib import contextmanager

# id(subject) -> the token of the session that subject takes part in now. The session's
# frame holds its subject, so the id cannot go to another object while the entry exists.
_active_sessions = {}
_naming = threading.local()


class one_session_at_a_time:
    """``with one_session_at_a_time(subject):`` refuses a second session on ``subject``
    while its first one runs. Make a new guard for each session: a guard is used once.

    There is no lock. The claim is ``dict.setdefault`` with an integer key, which is
    one step of the interpreter: no other thread, no garbage collection and no code of
    a subject runs inside it. So a claim cannot wait, and two sessions cannot both
    claim one subject. Only the session that claimed an entry removes it.

    Interrupts: one in the session body, in the claim or inside the release frees the
    subject. One in the guard's enter after the claim frees it when the interrupted
    call's exception and this guard are dropped. Limit: one that arrives after the
    session body ends and before the release starts leaves the subject reserved (there
    is no step that can undo a claim for another caller); a separate subject works.
    """

    def __init__(self, subject):
        self._session = _session(subject)
        self._used = False

    def __enter__(self):
        if self._used:      # a second entry would end the first session's claim
            raise RuntimeError('This guard is already in use. Make a new one_session_at_a_time '
                               'guard for each session.')
        self._used = True
        return self._session.__enter__()

    def __exit__(self, *error):
        return self._session.__exit__(*error)


@contextmanager
def _session(subject):
    key, token = id(subject), object()
    try:    # the claim and the session body are in one try, so every way out releases
        if _active_sessions.setdefault(key, token) is not token:
            raise RuntimeError(
                f'Subject {_name(subject)} is already taking part in a session. A subject '
                'takes part in one session at a time, because its recording and task state '
                'belong to that session. Wait for the running session to finish, or build a '
                'separate subject for each session that runs at the same time.')
        yield
    finally:
        try:
            _release(key, token)
        except BaseException:   # an interrupt inside the release: finish it, then pass it on
            _release(key, token)
            raise


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
