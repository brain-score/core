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
