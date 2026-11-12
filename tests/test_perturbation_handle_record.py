"""A typed perturbation response is tracked without driver metadata."""
from brainscore_core._evaluation.measures import record_measures
from brainscore_core._sessions.buffered import apply_state_change
from brainscore_core.contract import Subject
from brainscore_core.events import Perturbation, PerturbationApplied, Selection, StateChange
from brainscore_core.streaming import StreamEvent


class _TypedSubject(Subject):
    """Answers on `perturbation` with a typed payload and default metadata."""
    identifier = 'typed'
    in_channels = {'lesion'}
    out_channels = {'perturbation'}

    def participate(self, session):
        change = session.next_input().payload
        handle = change.handle_id or 'h1'
        session.respond(StreamEvent(
            channel='perturbation', t_ms=0.0,
            payload=PerturbationApplied(handle, change.target, change.perturbation)))


def _lesion():
    return StateChange(kind='ablation', target=Selection(layer='x'),
                       perturbation=Perturbation(kind='zero'))


def test_typed_payload_handle_is_recorded_and_undone_by_the_request():
    subject = _TypedSubject()
    with record_measures() as record:
        applied = apply_state_change(subject, _lesion())
        assert record.unreleased == {applied.handle_id}
        apply_state_change(subject, StateChange(kind='reset', handle_id=applied.handle_id))
    assert record.unreleased == set()


def _change_event(change, channel='lesion'):
    return StreamEvent(channel=channel, payload=change, t_ms=0.0)


def _typed(handle):
    return StreamEvent(channel='perturbation', t_ms=0.0,
                       payload=PerturbationApplied(handle, Selection(layer='x'), Perturbation(kind='zero')))


def test_delayed_typed_responses_are_tied_to_their_own_request():
    from brainscore_core.streaming import Session
    events = [_change_event(_lesion()), _change_event(_lesion()),
              _change_event(StateChange(kind='reset', handle_id='h1'))]
    session = Session(events, ['perturbation'])
    with record_measures() as record:
        for _ in events:
            session.next_input()
        for handle in ('h1', 'h2', 'h1'):
            session.respond(_typed(handle))
    assert record.applied == {'h1', 'h2'}
    assert record.undone == {'h1'}
    assert record.unreleased == {'h2'}


def test_response_without_handle_is_refused_when_the_request_is_ambiguous():
    import pytest
    from brainscore_core.streaming import Session
    events = [_change_event(_lesion()), _change_event(_lesion())]
    session = Session(events, ['perturbation'])
    for _ in events:
        session.next_input()
    with pytest.raises(ValueError, match='cannot tell which'):
        session.respond(StreamEvent(channel='perturbation', t_ms=0.0, payload=None))


def _family_subject(family):
    class _Subject(Subject):
        identifier = f'only-{family}'
        in_channels = {family}
        out_channels = {'perturbation'}
        active = set()

        def participate(self, session):
            session.declare_inputs(self.in_channels)
            change = session.next_input().payload
            if change.kind == 'reset':
                self.active.discard(change.handle_id)
                handle = change.handle_id
            else:
                handle = 'h-' + family
                self.active.add(handle)
            session.respond(StreamEvent(channel='perturbation', t_ms=0.0, payload=PerturbationApplied(
                handle, change.target, change.perturbation)))
    return _Subject()


def test_handle_only_reset_goes_on_the_family_the_change_was_applied_on():
    for family in ('stimulation', 'pharmacological'):
        subject = _family_subject(family)
        change = StateChange(kind=family, target=Selection(layer='x'),
                             perturbation=Perturbation(kind='zero'))
        with record_measures() as record:
            applied = apply_state_change(subject, change)
            assert subject.active == {applied.handle_id}
            apply_state_change(subject, StateChange(kind='reset', handle_id=applied.handle_id))
        assert subject.active == set()
        assert record.unreleased == set()
