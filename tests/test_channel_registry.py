import numpy as np

from brainscore_core import io_catalog
from brainscore_core.io_catalog import INPUT, OUTPUT, CatalogEntry


def test_registry_version_and_seed_entry_fields_are_queryable():
    assert io_catalog.REGISTRY_VERSION == "2.0"

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


def test_channels_can_be_listed_and_filtered_by_direction():
    assert "vision" in io_catalog.channels()
    assert "vision" in io_catalog.channels(INPUT)
    assert "neural" not in io_catalog.channels(INPUT)
    assert "neural" in io_catalog.channels(OUTPUT)
    assert "vision" not in io_catalog.channels(OUTPUT)

    assert all(entry.direction == INPUT for entry in io_catalog.by_direction(INPUT))
    assert all(entry.direction == OUTPUT for entry in io_catalog.by_direction(OUTPUT))


def test_validate_accepts_valid_payload_and_direction():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    neural = np.zeros((10, 5), dtype=np.float32)

    assert io_catalog.validate("vision", image, direction=INPUT) == []
    assert io_catalog.validate("neural:IT", neural, direction=OUTPUT) == []


def test_validate_reports_unknown_channel():
    failures = io_catalog.validate("telepathy", object(), direction=INPUT)

    assert len(failures) == 1
    assert "No Input/Output Catalog entry" in failures[0]


def test_validate_reports_wrong_direction():
    image = np.zeros((224, 224, 3), dtype=np.uint8)

    failures = io_catalog.validate("vision", image, direction=OUTPUT)

    assert any("not requested 'output'" in failure for failure in failures)


def test_validate_reports_declared_shape_mismatch():
    image = np.zeros((224, 224), dtype=np.uint8)

    failures = io_catalog.validate("vision", image, direction=INPUT)

    assert any("ndim" in failure for failure in failures)


def test_validate_reports_invalid_address_grammar():
    failures = io_catalog.validate("neural:", np.zeros(10), direction=OUTPUT)

    assert len(failures) == 1
    assert "invalid channel name" in failures[0]


def test_validate_reports_unaddressable_channel_family():
    failures = io_catalog.validate("text:left", "a sentence", direction=INPUT)

    assert any("not addressable" in failure for failure in failures)


def test_registration_keeps_builtin_column_ownership(monkeypatch):
    import pytest
    from brainscore_core.extensions import register_channel

    monkeypatch.setattr(io_catalog, '_CATALOG', dict(io_catalog._CATALOG))
    monkeypatch.setattr(io_catalog, '_STIMULUS_COLUMNS', {})
    expected = {
        'image_file_name': 'vision', 'image_path': 'vision', 'filename': 'vision',
        'sentence': 'text', 'text': 'text', 'video_path': 'vision',
        'audio_path': 'audio', 'audio_file_name': 'audio', 'audio_file': 'audio',
        'vision': None, 'image': None, 'audio': None, 'video': None,
    }
    for index, (column, owner) in enumerate(expected.items()):
        entry = CatalogEntry(f'column_probe_{index}', INPUT, 'StimulusSet', 'test', 'test')
        if owner is None:
            register_channel(entry, columns=[column])
            assert io_catalog.stimulus_columns()[column] == entry.name
        else:
            with pytest.raises(ValueError, match=f"already belongs to '{owner}'"):
                register_channel(entry, columns=[column])
            assert not io_catalog.has(entry.name)
            register_channel(entry, columns=[column], replace=True)
            assert io_catalog.stimulus_columns()[column] == entry.name
