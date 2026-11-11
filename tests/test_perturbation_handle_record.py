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
