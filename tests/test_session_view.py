"""Contract v3 rule 5.5: a stream the subject cannot take in never enters it."""

from brainscore_core.session_view import DeclaredInputSession
from brainscore_core.streaming import BufferedSession, StreamEvent


def test_view_forwards_completion_responses_and_collection():
    inner = BufferedSession([StreamEvent('vision', 'x', 0.0), StreamEvent('audio', 'y', 0.0)],
                            requested_output_channels=['neural:IT'], time_bins=[(0, 10)])
    view = DeclaredInputSession(inner, {'vision'})
    assert view.complete is inner.complete
    assert view.time_bins == [(0, 10)]
    assert view.requested_output_channels == ('neural:IT',)
    assert view.next_input().channel == 'vision'
    assert view.next_input() is None
    view.respond(StreamEvent('neural:IT', 'out', 0.0))
    assert inner.emitted[0].payload == 'out'
    assert view.collect('neural:IT') == inner.collect('neural:IT')


def test_view_carries_measurement_settings():
    inner = BufferedSession([StreamEvent('vision', 'x', 0.0)], requested_output_channels=['neural:IT'])
    inner.settings = {'number_of_trials': 3}
    assert DeclaredInputSession(inner, {'vision'}).settings == {'number_of_trials': 3}
