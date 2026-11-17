"""Tests for the v1.5 Input/Output Catalog.

The catalog documents the allowable inputs and outputs layered over the typed
events. It is documentation plus an optional best-effort shape check, not a
dispatch mechanism.
"""
import numpy as np
import pytest

from brainscore_core import io_catalog
from brainscore_core.io_catalog import CatalogEntry, INPUT, OUTPUT


class TestSeedCatalog:
    def test_expected_families_present(self):
        names = {e.name for e in io_catalog.all_entries()}
        expected = {
            "vision", "text", "audio", "instruction",
            "stimulation", "lesion", "observation", "proprioception",
            "neural", "behavior", "motor",
        }
        assert expected <= names

    def test_inputs_and_outputs_partition(self):
        inputs = {e.name for e in io_catalog.all_entries() if e.direction == INPUT}
        outputs = {e.name for e in io_catalog.all_entries() if e.direction == OUTPUT}
        assert "vision" in inputs and "lesion" in inputs
        assert {"neural", "behavior", "motor"} <= outputs
        assert inputs.isdisjoint(outputs)

    def test_every_entry_is_documented(self):
        for e in io_catalog.all_entries():
            assert e.direction in (INPUT, OUTPUT)
            assert e.payload_contract, f"{e.name} has no payload contract"
            assert e.handled_by, f"{e.name} has no handler"

    def test_motor_lists_policy_response_metadata(self):
        keys = io_catalog.get("motor").meta_keys
        assert {"answer", "valid", "parse_error", "fallback_action", "provenance"} <= set(keys)


class TestLookup:
    def test_plain_label_resolves(self):
        assert io_catalog.get("vision").name == "vision"

    def test_addressed_label_resolves_to_family(self):
        assert io_catalog.get("neural:IT").name == "neural"
        assert io_catalog.get("behavior:choice").name == "behavior"

    @pytest.mark.parametrize("family", ["lesion", "stimulation", "pharmacological"])
    def test_intervention_channels_are_not_addressable(self, family):
        from brainscore_core.events import StateChange
        assert io_catalog.get(family).addressing is None
        assert io_catalog.validate(family, StateChange(kind="ablation"),
                                   direction=io_catalog.INPUT) == []
        problems = io_catalog.validate(f"{family}:blocks.10", StateChange(kind="ablation"),
                                       direction=io_catalog.INPUT)
        assert any("not addressable" in problem for problem in problems)

    def test_has(self):
        assert io_catalog.has("neural:IT")
        assert io_catalog.has("vision")
        assert not io_catalog.has("telepathy")

    def test_unknown_label_raises(self):
        with pytest.raises(KeyError):
            io_catalog.get("telepathy")


class TestCheckPayload:
    def test_valid_vision_payload_has_no_warnings(self):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        assert io_catalog.check_payload("vision", img) == []

    def test_wrong_ndim_warns(self):
        bad = np.zeros((224, 224), dtype=np.uint8)  # missing channel axis
        warnings = io_catalog.check_payload("vision", bad)
        assert any("ndim" in w for w in warnings)

    def test_wrong_dtype_warns(self):
        bad = np.zeros((224, 224, 3), dtype=np.float32)
        warnings = io_catalog.check_payload("vision", bad)
        assert any("dtype" in w for w in warnings)

    def test_neural_accepts_1d_and_2d(self):
        assert io_catalog.check_payload("neural:IT", np.zeros(100)) == []
        assert io_catalog.check_payload("neural:IT", np.zeros((100, 5))) == []

    def test_non_array_payload_is_not_checked(self):
        # text payload is a str; no ndim/dtype, so no warnings.
        assert io_catalog.check_payload("text", "a sentence") == []


class TestRegister:
    def test_register_and_get_roundtrip(self, monkeypatch):
        monkeypatch.setattr(io_catalog, "_CATALOG", dict(io_catalog._CATALOG))
        entry = CatalogEntry(
            name="pupil", direction=OUTPUT, carried_by="start_recording('pupil')",
            payload_contract="(time,) float diameter in mm",
            handled_by="an eye-tracker harness",
        )
        io_catalog.register_channel(entry, replace=True)  # 'pupil' is a seeded family
        assert io_catalog.get("pupil") is entry
        assert io_catalog.get("pupil:left").name == "pupil"

    def test_register_rejects_silent_overwrite(self, monkeypatch):
        monkeypatch.setattr(io_catalog, "_CATALOG", dict(io_catalog._CATALOG))
        original = io_catalog.get("pupil")
        clash = CatalogEntry(
            name="pupil", direction=OUTPUT, carried_by="x", payload_contract="x",
            handled_by="another harness",
        )
        with pytest.raises(ValueError, match="already registered"):
            io_catalog.register_channel(clash)
        assert io_catalog.get("pupil") is original
        io_catalog.register_channel(original)  # same object: allowed, no change


class TestPayloadTypes:
    """Contract v3 rule 3.2: a family declares its payload type; a mismatch fails validation."""

    def test_seeded_families_declare_payload_types(self):
        from pathlib import Path
        import numpy as np
        from brainscore_core.events import StateChange
        assert io_catalog.get("text").payload_type is str
        vision = io_catalog.get("vision").payload_type
        assert np.ndarray in vision and str in vision and Path in vision
        assert io_catalog.get("lesion").payload_type is StateChange
        assert io_catalog.get("observation").payload_type is None  # harness-defined
        assert io_catalog.get("pupil").payload_type is None  # unchecked families stay unchecked

    def test_payload_type_notes_report_a_mismatch_and_nothing_else(self):
        assert io_catalog.payload_type_notes("text", "a sentence") == []
        assert io_catalog.payload_type_notes("vision:left", "/tmp/a.png") == []
        assert io_catalog.payload_type_notes("pupil", object()) == []
        notes = io_catalog.payload_type_notes("text", 42)
        assert notes == ["text: payload is int, not str (a Python str)"]

    def test_type_mismatch_is_a_validate_failure(self):
        assert io_catalog.validate("text", 42, direction=INPUT) == [
            "text: payload is int, not str (a Python str)"]
