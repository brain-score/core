"""A session's notes reach the score only when its participation returns without an error.

The ``Subject`` base stages what a session notes (its delivery, responses, provenance and
participant) and passes it to the score when the outermost participation of that session
returns. A failure drops all of it, apart from the intervention counts: a change that a
failed session left active is still seen by the release check.
"""
import pytest

from brainscore_core._evaluation.measures import note_session, record_measures
from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import Subject
from brainscore_core.events import Perturbation, PerturbationApplied, Selection, StateChange
from brainscore_core.metrics import Score
from brainscore_core.score_metadata import stamp_score_metadata
from brainscore_core.streaming import Session, StreamEvent


class _Text(Subject):
    """Answers each text input; ``fail_after`` answers, then refuses."""
    in_channels = {'text'}
    out_channels = {'behavior:text'}

    def __init__(self, identifier='inner', fail_after=None):
        self._identifier, self.fail_after = identifier, fail_after

    @property
    def identifier(self):
        return self._identifier

    def participate(self, session):
        answered = 0
        while (event := session.next_input()) is not None:
            if answered == self.fail_after:
                raise ExecutionNotFeasibleError('this subject cannot answer the second trial')
            session.respond(StreamEvent('behavior:text', 'word', event.t_ms, {}))
            answered += 1


class _Wrapper(Subject):
    """Notes its registration, delegates the same session, then ``fails`` in its cleanup."""
    in_channels = {'text'}
    out_channels = {'behavior:text'}

    def __init__(self, inner, identifier='wrapper', fails=False):
        self._inner, self._identifier, self.fails = inner, identifier, fails

    @property
    def identifier(self):
        return self._identifier

    def participate(self, session):
        note_session(session, provenance={'subject': self._identifier, 'registration': {'k': 1}})
        try:
            self._inner.participate(session)
        finally:
            if self.fails:
                raise RuntimeError('cleanup failed')


def _session(n=2):
    return Session([StreamEvent('text', 'a', float(i)) for i in range(n)], ['behavior:text'])


def _stamp(measures, subject_identifier='scored'):
    class _Scored:
        identifier = subject_identifier
        in_channels, traits = {'text'}, set()
    return stamp_score_metadata(Score(0.5), _Scored(), requested_channels={'behavior:text'},
                                entry_point='test', measures=measures).attrs


def _nothing_of_the_session(attrs):
    assert attrs['response'] == ()
    assert attrs['delivery'] is None
    assert attrs['provenance'] == ({'subject': 'scored', 'parts': 'not stated',
                                    'activation_store': 'not stated'},)


def test_a_session_refused_before_any_response_leaves_nothing():
    session = _session()
    with record_measures(subject='scored') as measures:
        with pytest.raises(ExecutionNotFeasibleError):
            _Wrapper(_Text(fail_after=0)).participate(session)
    _nothing_of_the_session(_stamp(measures))


def test_a_session_that_fails_after_a_response_leaves_nothing():
    session = _session()
    with record_measures(subject='scored') as measures:
        with pytest.raises(ExecutionNotFeasibleError):
            _Text(fail_after=1).participate(session)
    assert len(session.emitted) == 1           # the first answer was accepted
    _nothing_of_the_session(_stamp(measures))


def test_a_session_that_fails_in_cleanup_leaves_nothing():
    session = _session()
    with record_measures(subject='scored') as measures:
        with pytest.raises(RuntimeError, match='cleanup failed'):
            _Wrapper(_Text(), fails=True).participate(session)
    assert len(session.emitted) == 2
    _nothing_of_the_session(_stamp(measures))


def test_through_a_wrapper_around_a_wrapper_a_late_failure_leaves_nothing():
    subject = _Wrapper(_Wrapper(_Text(), identifier='middle'), identifier='outer', fails=True)
    with record_measures(subject='scored') as measures:
        with pytest.raises(RuntimeError, match='cleanup failed'):
            subject.participate(_session())
    _nothing_of_the_session(_stamp(measures))


def test_a_failed_attempt_then_a_good_one_stamps_only_the_good_one():
    with record_measures(subject='scored') as measures:
        with pytest.raises(ExecutionNotFeasibleError):
            _Wrapper(_Text(fail_after=1), identifier='first').participate(
                Session(iter([StreamEvent('text', 'a', 0.0), StreamEvent('text', 'b', 1.0)]),
                        ['behavior:text']))
        _Wrapper(_Text(), identifier='second').participate(_session())
    attrs = _stamp(measures)
    assert attrs['delivery'] == 'listed'
    subjects = {entry['subject'] for entry in attrs['provenance']}
    assert 'second' in subjects and 'first' not in subjects


def test_a_good_session_through_nested_wrappers_is_noted_once_under_the_outer_subject():
    subject = _Wrapper(_Wrapper(_Text(), identifier='middle'), identifier='outer')
    with record_measures(subject='scored') as measures:
        subject.participate(_session())
    attrs = _stamp(measures)
    assert attrs['delivery'] == 'listed'
    assert len(attrs['response']) == 1
    registrations = [entry for entry in attrs['provenance'] if 'registration' in entry]
    assert sorted(entry['subject'] for entry in registrations) == ['middle', 'outer']
    silent = [entry['subject'] for entry in attrs['provenance'] if entry.get('parts') == 'not stated']
    assert silent == ['outer']                 # the subject the session was given to; not 'scored'


class _Lesioned(Subject):
    """Applies a lesion it never undoes, then fails."""
    identifier = 'lesioned'
    in_channels = {'lesion'}
    out_channels = {'perturbation'}

    def participate(self, session):
        change = session.next_input().payload
        session.respond(StreamEvent('perturbation', PerturbationApplied(
            'h1', change.target, change.perturbation), 0.0))
        raise RuntimeError('failed after the change')


def _lesion_session():
    change = StateChange(kind='ablation', target=Selection(layer='x'), perturbation=Perturbation(kind='zero'))
    return Session([StreamEvent('lesion', change, 0.0)], ['perturbation'])


@pytest.mark.parametrize('nested_score', [False, True], ids=['one score', 'inside a nested score'])
def test_a_failed_session_still_passes_its_intervention_counts(nested_score):
    with record_measures(subject='scored') as measures:
        with pytest.raises(RuntimeError, match='failed after the change'):
            if nested_score:
                with record_measures(subject='lesioned'):
                    _Lesioned().participate(_lesion_session())
            else:
                _Lesioned().participate(_lesion_session())
    with pytest.raises(RuntimeError, match='left intervention'):
        measures.check_released('bench', 'scored')
    assert _stamp(measures)['response'] == ()


# --- interrupts and handle owners (review of 2026-11-19) --------------------------------

_FOUR = ('response', 'delivery', 'regions_per_pass', 'provenance')


def _four(measures):
    attrs = _stamp(measures)
    return {name: attrs[name] for name in _FOUR}


def _stream(n=2):
    return Session(iter([StreamEvent('text', 'a', float(i)) for i in range(n)]), ['behavior:text'])


def _good_only():
    with record_measures(subject='scored') as measures:
        _Wrapper(_Text(), identifier='good').participate(_session())
    return _four(measures)


@pytest.mark.parametrize('interrupt', [KeyboardInterrupt, SystemExit])
def test_an_interrupt_in_the_release_of_the_subject_leaves_nothing_of_the_session(monkeypatch, interrupt):
    """The session counts as failed when the release of its subject raises: the release is
    part of the participation, so nothing of that session reaches the next good score."""
    from brainscore_core._sessions import one_at_a_time
    release, calls = one_at_a_time._release, []

    def _interrupted_once(key, token):   # the test seam: the release itself is interrupted
        calls.append(key)
        if len(calls) == 1:
            raise interrupt()
        release(key, token)
    monkeypatch.setattr(one_at_a_time, '_release', _interrupted_once)
    failed = _Text(identifier='failed')
    with record_measures(subject='scored') as measures:
        session = _stream()
        with pytest.raises(interrupt):
            failed.participate(session)
        assert len(session.emitted) == 2       # the answers were accepted
        _Wrapper(_Text(), identifier='good').participate(_session())
    monkeypatch.setattr(one_at_a_time, '_release', release)
    assert _four(measures) == _good_only()
    failed.participate(_session())             # the subject was released


def _sweep(scenario):
    """Run ``scenario(arm, disarm)`` once per line that the measure record runs between
    ``arm()`` (the subject calls it as its body returns) and ``disarm()`` (the scenario calls
    it when the participation is over), with one interrupt at that line. Yields the line and
    the result of each interrupted run (the line is where the interrupt was raised)."""
    import sys
    from brainscore_core._evaluation import measures as module
    state = {'at': None, 'seen': 0, 'on': False}

    def line(frame, event, _):
        if state['on'] and event == 'line':
            state['seen'] += 1
            if state['seen'] == state['at']:
                state['on'] = False
                state['where'] = frame.f_lineno
                raise KeyboardInterrupt()
        return line

    def tracer(frame, event, _):
        return line if frame.f_code.co_filename == module.__file__ else None

    def arm():
        state.update(seen=0, on=True)
        sys.settrace(tracer)

    def disarm():
        state['on'] = False
        sys.settrace(None)
    try:
        scenario(arm, disarm)
    finally:
        disarm()
    total = state['seen']
    assert total > 0
    for at in range(1, total + 1):
        state['at'] = at
        state['where'] = None
        try:
            result = scenario(arm, disarm)
        finally:
            disarm()
        yield f"measures.py line {state['where']}", result


def _maybe_nested(nested_score):
    from contextlib import nullcontext
    return record_measures(subject='inner-score') if nested_score else nullcontext()


@pytest.mark.parametrize('nested_score', [False, True], ids=['one score', 'inside a nested score'])
def test_an_interrupt_anywhere_in_the_end_of_a_record_keeps_an_accepted_change_seen(nested_score):
    class _Applies(Subject):
        identifier = 'applies'
        in_channels, out_channels = {'lesion'}, {'perturbation'}

        def __init__(self, arm):
            self.arm, self.active = arm, set()

        def participate(self, session):
            change = session.next_input().payload
            session.respond(StreamEvent('perturbation', PerturbationApplied(
                'h1', change.target, change.perturbation), 0.0))
            self.active.add('h1')
            self.arm()

    def scenario(arm, disarm):
        subject = _Applies(arm)
        with record_measures(subject='scored') as measures:
            try:
                with _maybe_nested(nested_score):
                    subject.participate(_lesion_session())
            except KeyboardInterrupt:
                pass
            disarm()
        return subject.active, measures.unreleased

    for at, (active, unreleased) in _sweep(scenario):
        assert active == {'h1'} and unreleased == {'h1'}, f'an interrupt at {at} hid the change'


@pytest.mark.parametrize('nested_score', [False, True], ids=['one score', 'inside a nested score'])
def test_an_interrupt_anywhere_in_the_end_of_a_record_publishes_all_or_nothing(nested_score):
    def run(arm, disarm):
        class _Armed(_Text):
            def participate(self, session):
                super().participate(session)
                arm()
        with record_measures(subject='scored') as measures:
            try:
                with _maybe_nested(nested_score):
                    try:
                        _Wrapper(_Armed(identifier='inner-of-late'), identifier='late').participate(_stream())
                    finally:
                        disarm()   # limit: the first step of a score record's own exit is not swept
            except KeyboardInterrupt:
                pass
            _Wrapper(_Text(), identifier='good').participate(_session())
        return _four(measures)

    complete = run(lambda: None, lambda: None)
    allowed = (_good_only(), complete)
    for at, four in _sweep(run):
        assert four in allowed, f'an interrupt at {at} published part of a failed session'


class _Handles(Subject):
    """Applies the handle ``h1`` on a lesion and undoes it on a reset; ``fail_after`` raises
    after it answered."""
    in_channels, out_channels = {'lesion'}, {'perturbation'}

    def __init__(self, identifier, fail_after=False):
        self._identifier, self.fail_after, self.active = identifier, fail_after, set()

    @property
    def identifier(self):
        return self._identifier

    def participate(self, session):
        change = session.next_input().payload
        (self.active.discard if change.kind == 'reset' else self.active.add)('h1')
        session.respond(StreamEvent('perturbation', PerturbationApplied(
            'h1', change.target, change.perturbation), 0.0, {'kind': change.kind}))
        if self.fail_after:
            raise RuntimeError('failed after it answered')


def _change(kind):
    if kind == 'reset':
        return Session([StreamEvent('lesion', StateChange(kind='reset', handle_id='h1'), 0.0)],
                       ['perturbation'])
    return _lesion_session()


def test_an_undo_on_another_subject_does_not_cancel_a_change_with_the_same_handle():
    a, b = _Handles('a'), _Handles('b')
    b.participate(_change('lesion'))            # b holds its own h1 from before the score
    b.fail_after = True
    with record_measures(subject='scored') as measures:
        a.participate(_change('lesion'))
        with pytest.raises(RuntimeError, match='failed after it answered'):
            with record_measures(subject='b'):
                b.participate(_change('reset'))
    assert a.active == {'h1'} and b.active == set()
    with pytest.raises(RuntimeError, match='left intervention'):
        measures.check_released('bench', 'scored')


@pytest.mark.parametrize('route', ['separate sessions', 'through a wrapper', 'a failed retry',
                                   'nested scores'])
def test_an_undo_on_the_same_subject_releases_its_change(route):
    subject = _Handles('one')

    class _Delegates(Subject):
        identifier = 'wrapper'
        in_channels, out_channels = {'lesion'}, {'perturbation'}

        def participate(self, session):
            subject.participate(session)

    with record_measures(subject='scored') as measures:
        if route == 'through a wrapper':
            _Delegates().participate(_change('lesion'))
        elif route == 'a failed retry':
            subject.fail_after = True
            with pytest.raises(RuntimeError):
                subject.participate(_change('lesion'))
            subject.fail_after = False
        elif route == 'nested scores':
            with record_measures(subject='one'):
                subject.participate(_change('lesion'))
        else:
            subject.participate(_change('lesion'))
        subject.participate(_change('reset'))
    assert subject.active == set()
    measures.check_released('bench', 'scored')


def test_a_caller_can_still_name_a_session_that_no_subject_base_runs():
    """``note_session(session, subject=...)`` names a session for a caller that runs it
    itself; a ``Subject`` that takes part names its own session, and that name wins."""
    note_session(_session(), subject='custom')           # outside a score: no effect, no error
    with record_measures(subject='scored') as measures:
        by_caller = _session()
        note_session(by_caller, subject='custom')
        named_by_both = _session()
        note_session(named_by_both, subject='custom-too')
        _Text(identifier='taker').participate(named_by_both)
    subjects = {entry['subject'] for entry in _stamp(measures)['provenance']}
    assert {'custom', 'taker'} <= subjects and 'custom-too' not in subjects
