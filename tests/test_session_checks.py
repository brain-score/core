"""Rule 3.4 and 3.7: inputs and responses are checked against the channel catalog."""

import numpy as np
import pytest

from brainscore_core.streaming import (
    CONTROL_CHANNEL, FixedSource, Session, StreamEvent, begin_event,
)


def test_response_on_a_channel_that_was_not_requested_raises():
    session = Session([], requested_output_channels=['neural:IT'])
    with pytest.raises(ValueError, match='not requested'):
        session.respond(StreamEvent('neural:V4', np.zeros(3), 0.0))
    assert session.emitted == []


def test_a_bare_family_does_not_stand_for_a_requested_address():
    session = Session([], requested_output_channels=['neural:IT'])
    with pytest.raises(ValueError, match='not requested'):
        session.respond(StreamEvent('neural', np.zeros(3), 0.0))


def test_response_with_a_wrong_payload_type_raises():
    session = Session([], requested_output_channels=['neural:IT'])
    with pytest.raises(ValueError, match='payload is str'):
        session.respond(StreamEvent('neural:IT', 'not an array', 0.0))


def test_response_with_a_valid_payload_is_kept():
    session = Session([], requested_output_channels=['neural:IT'])
    session.respond(StreamEvent('neural:IT', np.zeros(3), 0.0))
    assert len(session.emitted) == 1


def test_a_lazy_source_is_checked_when_the_event_is_delivered():
    def events():
        yield StreamEvent('text', 'fine', 0.0)
        yield StreamEvent('text', 42, 1.0)
    session = Session(events())
    assert session.next_input().payload == 'fine'
    with pytest.raises(ValueError, match='payload is int'):
        session.next_input()


def test_an_eager_list_is_checked_at_construction():
    with pytest.raises(ValueError, match='payload is int'):
        Session([StreamEvent('text', 42, 0.0)])


def test_an_unknown_input_channel_raises():
    session = Session(iter([StreamEvent('not_a_family', 1, 0.0)]))
    with pytest.raises(ValueError, match='No Input/Output Catalog entry'):
        session.next_input()


def test_an_unknown_output_channel_raises():
    session = Session([], requested_output_channels=['not_a_family'])
    with pytest.raises(ValueError, match='No Input/Output Catalog entry'):
        session.respond(StreamEvent('not_a_family', 1, 0.0))


def test_a_bad_control_payload_raises_on_a_lazy_source():
    session = Session(iter([begin_event(), StreamEvent(CONTROL_CHANNEL, 'jump', 0.0)])).declare_inputs({'control'})
    assert session.next_input().payload == 'begin'
    with pytest.raises(ValueError, match="control payload must be"):
        session.next_input()


def test_a_session_with_no_requested_outputs_accepts_any_catalog_channel():
    session = Session([])
    session.respond(StreamEvent('neural:V4', np.zeros(3), 0.0))
    session.respond(StreamEvent('behavior', 'cat', 0.0))
    assert [event.channel for event in session.emitted] == ['neural:V4', 'behavior']


def test_a_checked_source_is_not_checked_again():
    source = FixedSource([StreamEvent('text', 'fine', 0.0)])
    assert source.checked
    assert Session(source).next_input().payload == 'fine'
