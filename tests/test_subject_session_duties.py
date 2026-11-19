"""The ``Subject`` base class runs the session duties around every ``participate``.

A minimal subject that only reads and answers gets them: an input on a channel it does
not declare never reaches its body, and a second session on the busy subject is
refused. A nested call with the same session (``super().participate``, a wrapper that
delegates the session) passes.
"""
import threading

import pytest

from brainscore_core.contract import Subject
from brainscore_core.streaming import Session, StreamEvent


class _Reader(Subject):
    """Reads every input and answers once; it calls nothing else."""
    identifier = 'reader'
    in_channels = {'text'}
    out_channels = {'behavior:text'}

    def __init__(self):
        self.seen, self.during = [], None

    def participate(self, session):
        while (event := session.next_input()) is not None:
            self.seen.append(event.channel)
            if self.during is not None:
                self.during()
        session.respond(StreamEvent('behavior:text', 'done', None, {}))


def _session(*channels):
    events = [StreamEvent(channel, 1.0 if channel == 'reward' else 'word', None, {})
              for channel in channels]
    return Session(events, outputs=('behavior:text',))


def test_an_undeclared_input_never_reaches_the_body():
    subject = _Reader()
    subject.participate(_session('reward', 'text', 'reward'))
    assert subject.seen == ['text']


def test_a_second_session_on_the_busy_subject_is_refused_with_a_remedy():
    subject, refusals = _Reader(), []

    def second():
        try:
            subject.participate(_session('text'))
        except RuntimeError as error:
            refusals.append(str(error))

    def during():
        subject.during = None           # only the first session starts another one
        second()

    subject.during = during
    subject.participate(_session('text'))
    assert len(refusals) == 1 and 'reader' in refusals[0] and 'separate subject' in refusals[0]
    subject.participate(_session('text'))       # the subject is free again


def test_a_session_from_another_thread_is_refused():
    subject, refusals = _Reader(), []

    def other_thread():
        try:
            subject.participate(_session('text'))
        except RuntimeError:
            refusals.append(True)

    def during():
        subject.during = None           # only the first session starts another one
        thread = threading.Thread(target=other_thread, daemon=True)
        thread.start()
        thread.join(timeout=10)

    subject.during = during
    subject.participate(_session('text'))
    assert refusals == [True]


def test_a_nested_call_with_the_same_session_passes():
    class Outer(_Reader):
        def participate(self, session):
            super().participate(session)

    class Delegating(Subject):
        identifier = 'delegating'
        in_channels = {'text', 'reward'}
        out_channels = {'behavior:text'}

        def __init__(self, inner):
            self.inner = inner

        def participate(self, session):
            self.inner.participate(session)

    outer = Outer()
    outer.participate(_session('reward', 'text'))
    assert outer.seen == ['text']
    inner = _Reader()
    Delegating(inner).participate(_session('reward', 'text'))
    assert inner.seen == ['text']          # each subject narrows to its own declaration


def test_a_subject_that_still_holds_the_guard_itself_is_told_to_delete_that_line():
    from brainscore_core._sessions.one_at_a_time import one_session_at_a_time

    class HoldsTheGuard(_Reader):
        def participate(self, session):
            with one_session_at_a_time(self):
                super().participate(session)

    with pytest.raises(RuntimeError, match='delete'):
        HoldsTheGuard().participate(_session('text'))



class _Reads:
    """A mixin that is not a Subject and supplies the subject's participate."""

    def participate(self, session):
        self.seen = [event.channel for event in iter(session.next_input, None)]
        if self.during is not None:
            self.during()


def _reward_then_text():
    return Session([StreamEvent('reward', 1.0, None, {}), StreamEvent('text', 'word', None, {})],
                   outputs=('behavior:text',))


@pytest.mark.parametrize('bases', [(_Reads, Subject), (_Reads, _Reader)],
                         ids=['mixin', 'mixin before a subject parent'])
def test_a_participate_from_a_mixin_gets_both_duties(bases):
    Mixed = type('Mixed', bases, {'identifier': 'mixed', 'in_channels': {'text'},
                                  'out_channels': {'behavior:text'}, 'during': None})
    subject, refusals = Mixed(), []

    def during():
        subject.during = None
        try:
            subject.participate(_session('text'))
        except RuntimeError:
            refusals.append(True)

    subject.during = during
    subject.participate(_reward_then_text())
    assert subject.seen == ['text'] and refusals == [True]


def test_a_decorated_participate_that_copies_the_parent_attributes_gets_both_duties():
    import functools

    class Decorated(_Reader):
        @functools.wraps(_Reader.participate)
        def participate(self, session):
            self.seen = [event.channel for event in iter(session.next_input, None)]

    subject = Decorated()
    subject.participate(_reward_then_text())
    assert subject.seen == ['text']


def test_a_participate_that_is_a_callable_object_gets_both_duties():
    class Reads:
        def __call__(self, session):
            self.seen = [event.channel for event in iter(session.next_input, None)]

    reads = Reads()
    subject = type('Odd', (_Reader,), {'participate': reads})()
    subject.participate(_reward_then_text())
    assert reads.seen == ['text']


def test_a_nested_call_while_the_inputs_are_declared_is_refused():
    class Reenters(_Reader):
        @property
        def in_channels(self):
            self.participate(self.session)
            return {'text'}

    subject = Reenters()
    subject.session = _reward_then_text()
    with pytest.raises(RuntimeError):
        subject.participate(subject.session)
    assert 'reward' not in subject.seen


def test_a_deferred_participate_is_refused_with_a_remedy():
    class Deferred(_Reader):
        def participate(self, session):
            yield session.next_input()

    subject = Deferred()
    with pytest.raises(TypeError, match='plain method'):
        subject.participate(_reward_then_text())
    with pytest.raises(TypeError, match='plain method'):    # the claim was released
        subject.participate(_session('text'))


def test_an_inner_subject_with_a_broader_declaration_does_not_reopen_an_input():
    class Wide(_Reader):
        in_channels = {'text', 'reward'}

    class NarrowOuter(Subject):
        identifier = 'narrow'
        in_channels = {'text'}
        out_channels = {'behavior:text'}

        def __init__(self, inner):
            self.inner = inner

        def participate(self, session):
            self.inner.participate(session)

    inner = Wide()
    NarrowOuter(inner).participate(_session('reward', 'text', 'reward'))
    assert inner.seen == ['text']


def test_a_failing_declaration_releases_the_subject():
    class FailsOnce(_Reader):
        fail = True

        @property
        def in_channels(self):
            if self.fail:
                raise KeyError('declaration')
            return {'text'}

    subject = FailsOnce()
    with pytest.raises(KeyError):
        subject.participate(_session('text'))
    subject.fail = False
    subject.participate(_reward_then_text())
    assert subject.seen == ['text']


def test_a_diamond_runs_the_body_that_python_selects():
    class Root:
        def participate(self, session):
            self.body = 'root'

    class Right(Root):
        def participate(self, session):
            self.body = 'right'
            self.seen = [event.channel for event in iter(session.next_input, None)]

    Left = type('Left', (Root, Subject), {})
    Leaf = type('Leaf', (Left, Right), {'identifier': 'leaf', 'in_channels': {'text'},
                                        'out_channels': {'behavior:text'}})
    subject = Leaf()
    subject.participate(_reward_then_text())
    assert subject.body == 'right' and subject.seen == ['text']


def test_a_participate_that_is_not_callable_is_refused_before_any_session():
    from brainscore_core.compatibility import check_channel_compatibility
    from types import SimpleNamespace

    NoMethod = type('NoMethod', (_Reader,), {'participate': None})
    benchmark = SimpleNamespace(identifier='b', required_input_channels={'text'},
                                requested_output_channels={'behavior:text'})
    with pytest.raises(TypeError, match='def participate'):
        check_channel_compatibility(NoMethod(), benchmark)


def test_a_mixin_whose_class_hook_does_not_call_super_cannot_suppress_the_duties():
    class Quiet:
        def __init_subclass__(cls, **kwargs):
            pass                            # does not call super()

        def participate(self, session):
            self.seen = [event.channel for event in iter(session.next_input, None)]

    Mixed = type('Mixed', (Quiet, Subject), {'identifier': 'quiet', 'in_channels': {'text'},
                                             'out_channels': {'behavior:text'}})
    subject = Mixed()
    subject.participate(_reward_then_text())
    assert subject.seen == ['text']


def test_a_participate_assigned_after_the_class_statement_gets_both_duties():
    class Late(_Reader):
        pass

    def late(self, session):
        self.seen = [event.channel for event in iter(session.next_input, None)]

    Late.participate = late
    subject = Late()
    subject.participate(_reward_then_text())
    assert subject.seen == ['text']
    on_instance = _Reader()
    on_instance.participate = lambda session: setattr(
        on_instance, 'seen', [event.channel for event in iter(session.next_input, None)])
    on_instance.participate(_reward_then_text())
    assert on_instance.seen == ['text']


def test_a_copied_or_pickled_subject_keeps_both_duties():
    import copy
    import pickle

    for other in (copy.copy(_Picklable()), copy.deepcopy(_Picklable()),
                  pickle.loads(pickle.dumps(_Picklable()))):
        other.participate(_reward_then_text())
        assert other.seen == ['text'] and vars(other).keys() == {'seen', 'during'}


class _Picklable(_Reader):
    """A top-level subject, so pickle can find its class."""


def test_a_subject_with_slots_or_a_fallback_getattr_keeps_both_duties():
    class Slotted(_Reader):
        __slots__ = ('extra',)

    class Forwards(_Reader):            # as torch.nn.Module: __getattr__ for missing names only
        def __getattr__(self, name):
            if name == 'layer':
                return 'from __getattr__'
            raise AttributeError(name)

    for subject in (Slotted(), Forwards()):
        subject.participate(_reward_then_text())
        assert subject.seen == ['text']
    assert Forwards().layer == 'from __getattr__'


def test_a_participate_from_the_getattr_fallback_gets_both_duties():
    class Fallback(_Reader):
        __slots__ = ('participate',)        # never set: the lookup falls back to __getattr__

        def __getattr__(self, name):
            if name == 'participate':
                return self._read
            raise AttributeError(name)

        def _read(self, session):
            while (event := session.next_input()) is not None:
                self.seen.append(event.channel)
            session.respond(StreamEvent('behavior:text', 'done', None, {}))

    subject = Fallback()
    subject.participate(_reward_then_text())
    assert subject.seen == ['text']


def test_a_cooperative_mixin_keeps_its_own_attribute_lookup():
    class GainMixin:
        def __getattribute__(self, name):
            if name == 'default_gain':
                return 3
            return super().__getattribute__(name)

    class Model(_Reader, GainMixin):
        pass

    subject = Model()
    assert subject.default_gain == 3
    subject.participate(_reward_then_text())
    assert subject.seen == ['text']


def test_an_interrupt_on_the_way_out_does_not_admit_the_old_session_later():
    import sys
    contract_file = sys.modules[Subject.__module__].__file__

    class Probe(_Reader):
        old = None
        refused_old = None

        def participate(self, session):
            if self.old is not None:        # a new session runs: try the old one from inside it
                try:
                    self.participate(self.old)
                    self.refused_old = False
                except RuntimeError:
                    self.refused_old = True

    judged = 0
    for at in range(1, 80):
        subject, first, seen, fired = Probe(), _session('text'), [0], []

        def trace(frame, event, argument):
            if frame.f_code.co_filename != contract_file:
                return None
            if event == 'line' and not fired:
                seen[0] += 1
                if seen[0] == at:
                    fired.append(True)
                    raise KeyboardInterrupt
            return trace

        previous = sys.gettrace()
        sys.settrace(trace)
        try:
            subject.participate(first)
        except KeyboardInterrupt:
            pass
        finally:
            sys.settrace(previous)
        if not fired:
            break
        subject.old = first
        try:
            subject.participate(_session('text'))
        except RuntimeError:
            continue        # the stated limit: the interrupt left the subject reserved
        judged += 1
        assert subject.refused_old, f'the old session entered a new one after an interrupt at line event {at}'
    assert judged >= 3
