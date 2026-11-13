import numpy as np

from brainscore_core import io_catalog
from brainscore_core.io_catalog import INPUT, OUTPUT, CatalogEntry


def test_seed_entry_fields_are_queryable():
    vision = io_catalog.get("vision")
    assert vision.kind == INPUT
    assert vision.direction == INPUT
    assert vision.owner == "core"
    assert vision.materializers == ("StimulusSet",)

    neural = io_catalog.get("neural:IT")
    assert neural.kind == OUTPUT
    assert neural.direction == OUTPUT
    assert neural.owner == "core"
    assert "NeuroidAssembly" in neural.materializers

    observation = io_catalog.get("observation")
    proprioception = io_catalog.get("proprioception")
    motor = io_catalog.get("motor")
    assert observation.owner == "environment harness"
    assert proprioception.owner == "environment harness"
    assert motor.owner == "environment harness"


def test_catalog_entry_keeps_v1_constructor_defaults():
    entry = CatalogEntry(
        name="pupil",
        kind=OUTPUT,
        carried_by="start_recording('pupil')",
        payload_contract="(time,) float diameter in mm",
        handled_by="an eye-tracker harness",
    )

    assert entry.direction == OUTPUT
    assert entry.shape_validator is None
    assert entry.owner == "core"
    assert entry.materializers == ()


def test_entries_split_by_direction():
    by_direction = {INPUT: set(), OUTPUT: set()}
    for entry in io_catalog.all_entries():
        by_direction[entry.direction].add(entry.name)
    assert "vision" in by_direction[INPUT]
    assert "neural" in by_direction[OUTPUT]
    assert "neural" not in by_direction[INPUT]
    assert "vision" not in by_direction[OUTPUT]


def test_validate_accepts_valid_payload_and_direction():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    neural = np.zeros((10, 5), dtype=np.float32)

    assert io_catalog.validate("vision:image", image, direction=INPUT) == []
    assert io_catalog.validate("neural:IT", neural, direction=OUTPUT) == []


def test_form_problem_accepts_named_forms_and_rejects_bare_vision():
    for channel in ("vision:image", "vision:video", "text", "neural:IT"):
        assert io_catalog.form_problem(channel) is None
    for channel in ("vision", "vision:still"):
        assert "names its form" in io_catalog.form_problem(channel)


def test_validate_reports_a_visual_channel_without_a_form():
    image = np.zeros((224, 224, 3), dtype=np.uint8)

    assert any("names its form" in failure
               for failure in io_catalog.validate("vision", image, direction=INPUT))


def test_validate_reports_unknown_channel():
    failures = io_catalog.validate("telepathy", object(), direction=INPUT)

    assert len(failures) == 1
    assert "No Input/Output Catalog entry" in failures[0]


def test_validate_reports_wrong_direction():
    image = np.zeros((224, 224, 3), dtype=np.uint8)

    failures = io_catalog.validate("vision:image", image, direction=OUTPUT)

    assert any("not requested 'output'" in failure for failure in failures)


def test_validate_reports_declared_shape_mismatch():
    image = np.zeros((224, 224), dtype=np.uint8)

    failures = io_catalog.validate("vision:image", image, direction=INPUT)

    assert any("ndim" in failure for failure in failures)


def test_validate_reports_invalid_address_grammar():
    failures = io_catalog.validate("neural:", np.zeros(10), direction=OUTPUT)

    assert len(failures) == 1
    assert "invalid channel name" in failures[0]


def test_validate_reports_unaddressable_channel_family():
    failures = io_catalog.validate("text:left", "a sentence", direction=INPUT)

    assert any("not addressable" in failure for failure in failures)


def test_registration_keeps_builtin_column_ownership(monkeypatch):
    """A column core already maps cannot be claimed by a new family without replace=True."""
    import pytest
    from brainscore_core.extensions import register_channel

    monkeypatch.setattr(io_catalog, '_CATALOG', dict(io_catalog._CATALOG))
    monkeypatch.setattr(io_catalog, '_STIMULUS_COLUMNS', dict(io_catalog._STIMULUS_COLUMNS))
    for index, (column, owner) in enumerate(io_catalog.stimulus_columns().items()):
        entry = CatalogEntry(f'column_probe_{index}', INPUT, 'StimulusSet', 'test', 'test')
        with pytest.raises(ValueError, match=f"already belongs to '{owner}'"):
            register_channel(entry, columns=[column])
        assert not io_catalog.has(entry.name)
        register_channel(entry, columns=[column], replace=True)
        assert io_catalog.column_channel(column) == entry.name
