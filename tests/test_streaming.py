import pytest

from brainscore_core.streaming import (
    InMemorySession,
    Session,
    StreamEvent,
    parse_channel,
)


def test_stream_event_construction():
    event = StreamEvent(channel="vision", payload={"id": "stim-0"}, t_ms=12.5)
    other = StreamEvent(channel="text", payload="caption", t_ms=13.0)

    assert event.channel == "vision"
    assert event.payload == {"id": "stim-0"}
    assert event.t_ms == 12.5
    assert event.meta == {}

    event.meta["presentation_id"] = "stim-0"
    assert other.meta == {}


@pytest.mark.parametrize(
    "name,expected",
    [
        ("vision", ("vision", None)),
        ("text", ("text", None)),
        ("skin_conductance", ("skin_conductance", None)),
        ("custom_channel1", ("custom_channel1", None)),
    ],
)
def test_parse_channel_valid_unaddressed(name, expected):
    assert parse_channel(name) == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("neural:IT", ("neural", "IT")),
        ("neural:language_network", ("neural", "language_network")),
        ("lesion:layer20[0:256]", ("lesion", "layer20[0:256]")),
        ("stimulation:language_model.layers.20", ("stimulation", "language_model.layers.20")),
    ],
)
def test_parse_channel_valid_addressed(name, expected):
    assert parse_channel(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "",
        "Vision",
        "bad-family",
        "bad family",
        ":IT",
        "neural:",
        "neural::IT",
        "neural:IT:",
        "neural:bad address",
        "neural:bad!",
    ],
)
def test_parse_channel_invalid_names(name):
    with pytest.raises(ValueError):
        parse_channel(name)


def test_parse_channel_rejects_non_string():
    with pytest.raises(TypeError):
        parse_channel(None)


def test_session_minimal_contract_excludes_collect():
    assert "next_input" in Session.__dict__
    assert "emit" in Session.__dict__
    assert "collect" not in Session.__dict__


def test_in_memory_session_drives_inputs_and_records_emits():
    inputs = [
        StreamEvent(channel="vision", payload="image-0", t_ms=0.0),
        StreamEvent(channel="instruction", payload="choose", t_ms=5.0),
    ]
    session = InMemorySession(inputs)

    assert session.next_input() is inputs[0]
    assert session.next_input() is inputs[1]
    assert session.next_input() is None

    output = StreamEvent(channel="behavior", payload={"choice": "cat"}, t_ms=12.0)
    session.emit(output)

    assert session.emitted == [output]
