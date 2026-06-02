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
            "vision", "text", "audio", "video", "instruction",
            "stimulation", "lesion", "observation", "proprioception",
            "neural", "behavior", "motor",
        }
        assert expected <= names

    def test_inputs_and_outputs_partition(self):
        inputs = {e.name for e in io_catalog.inputs()}
        outputs = {e.name for e in io_catalog.outputs()}
        assert "vision" in inputs and "lesion" in inputs
        assert {"neural", "behavior", "motor"} <= outputs
        assert inputs.isdisjoint(outputs)

    def test_every_entry_is_documented(self):
        for e in io_catalog.all_entries():
            assert e.kind in (INPUT, OUTPUT)
            assert e.payload_contract, f"{e.name} has no payload contract"
            assert e.handled_by, f"{e.name} has no handler"


class TestLookup:
    def test_plain_label_resolves(self):
        assert io_catalog.get("vision").name == "vision"

    def test_addressed_label_resolves_to_family(self):
        assert io_catalog.get("neural:IT").name == "neural"
        assert io_catalog.get("lesion:V4").name == "lesion"
        assert io_catalog.get("stimulation:layer20").name == "stimulation"

    def test_has(self):
        assert io_catalog.has("neural:IT")
        assert io_catalog.has("vision")
        assert not io_catalog.has("telepathy")

    def test_unknown_label_raises(self):
        with pytest.raises(KeyError):
            io_catalog.get("telepathy")

    def test_describe_returns_text(self):
        text = io_catalog.describe("neural:IT")
        assert "neural" in text and "carried by" in text


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
    def test_register_and_get_roundtrip(self):
        entry = CatalogEntry(
            name="pupil", kind=OUTPUT, carried_by="start_recording('pupil')",
            payload_contract="(time,) float diameter in mm",
            handled_by="an eye-tracker harness",
        )
        io_catalog.register(entry)
        assert io_catalog.get("pupil") is entry
        assert io_catalog.get("pupil:left").name == "pupil"
