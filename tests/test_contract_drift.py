"""Drift guard for the UMI v2.0 public contract surface.

This fixture is the in-repo source of truth for docs drift checks. Contract
changes must update both this pinned list and the external v2 docs:
API-CONTRACT.md and _DESIGN-CANON.md in the unified-model-interface-v2 vault.
"""

from dataclasses import MISSING, fields
import inspect

import pytest

from brainscore_core import io_catalog
from brainscore_core.contract import Subject
from brainscore_core.streaming import Session, StreamEvent


PINNED_SEED_CHANNELS = {
    "audio": (io_catalog.INPUT, "core"),
    "behavior": (io_catalog.OUTPUT, "core"),
    "instruction": (io_catalog.INPUT, "core"),
    "lesion": (io_catalog.INPUT, "core"),
    "motor": (io_catalog.OUTPUT, "environment harness"),
    "neural": (io_catalog.OUTPUT, "core"),
    "observation": (io_catalog.INPUT, "environment harness"),
    "pharmacological": (io_catalog.INPUT, "core"),
    "proprioception": (io_catalog.INPUT, "environment harness"),
    "pupil": (io_catalog.OUTPUT, "human harness"),
    "skin_conductance": (io_catalog.OUTPUT, "human harness"),
    "stimulation": (io_catalog.INPUT, "core"),
    "text": (io_catalog.INPUT, "core"),
    "video": (io_catalog.INPUT, "core"),
    "vision": (io_catalog.INPUT, "core"),
}


class _MinimalSubject(Subject):
    @property
    def identifier(self):
        return "minimal-subject"

    @property
    def region_layer_map(self):
        return {}

    @property
    def supported_modalities(self):
        return set()

    def process(self, input_event):
        return input_event


def test_seed_channel_registry_matches_pinned_contract_surface():
    actual = {
        entry.name: (entry.direction, entry.owner)
        for entry in io_catalog.all_entries()
    }

    assert actual == PINNED_SEED_CHANNELS


def test_registry_is_runtime_queryable_and_entries_are_well_formed():
    assert io_catalog.REGISTRY_VERSION
    assert set(io_catalog.channels()) == set(PINNED_SEED_CHANNELS)
    assert set(io_catalog.channels(io_catalog.INPUT)) == {
        name
        for name, (direction, _owner) in PINNED_SEED_CHANNELS.items()
        if direction == io_catalog.INPUT
    }
    assert set(io_catalog.channels(io_catalog.OUTPUT)) == {
        name
        for name, (direction, _owner) in PINNED_SEED_CHANNELS.items()
        if direction == io_catalog.OUTPUT
    }

    for entry in io_catalog.all_entries():
        assert entry.direction in {
            io_catalog.INPUT,
            io_catalog.OUTPUT,
            io_catalog.BOTH,
        }
        assert entry.owner
        assert entry.payload_contract
        assert io_catalog.get(entry.name) is entry


def test_stream_event_field_contract_is_pinned():
    stream_fields = fields(StreamEvent)

    assert [field.name for field in stream_fields] == [
        "channel",
        "payload",
        "t_ms",
        "meta",
    ]
    assert stream_fields[3].default_factory is dict
    assert all(field.default is MISSING for field in stream_fields[:3])


def test_session_abc_contract_is_minimal():
    assert Session.__abstractmethods__ == {"next_input", "emit"}
    assert not hasattr(Session, "collect")


def test_subject_interact_and_reset_contract_are_pinned():
    assert hasattr(Subject, "interact")
    assert hasattr(Subject, "reset")
    assert "interact" not in Subject.__abstractmethods__
    assert "reset" not in Subject.__abstractmethods__

    subject = _MinimalSubject()
    with pytest.raises(NotImplementedError, match="no v2 interact"):
        subject.interact(session=None)
    assert subject.reset() is None


def test_typed_helpers_are_importable():
    from brainscore_core.streaming_helpers import (
        apply_state_change,
        run_environment,
        score_behavior,
        score_stimuli,
    )

    for helper in (
        apply_state_change,
        run_environment,
        score_behavior,
        score_stimuli,
    ):
        assert callable(helper)
        assert inspect.isfunction(helper)
