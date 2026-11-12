import os
import json
from pathlib import Path
import tempfile
import pytest
import subprocess
from unittest.mock import Mock, patch, ANY

from brainscore_core.plugin_management.handle_metadata import (
    validate_metadata_file,
    create_metadata_pr,
)


class TestHandleMetadata:
    def test_validate_metadata_file_valid_models(self):
        """Test that a valid metadata file for models passes validation."""
        valid_yaml = """
                    models:
                      test-model:
                        architecture: "CNN"
                        model_family: null
                        total_parameter_count: 1000000
                        trainable_parameter_count: 900000
                        total_layers: 50
                        trainable_layers: 20
                        model_size_mb: 120.5
                        training_dataset: "imagenet"
                        task_specialization: "classification"
                        brainscore_link: "https://example.com"
                        huggingface_link: null
                        extra_notes: null
                        runnable: False
                    """
        with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
            tmp.write(valid_yaml)
            tmp_path = tmp.name

        errors, data = validate_metadata_file(tmp_path)
        os.remove(tmp_path)
        assert errors == []
        assert "models" in data
        assert "test-model" in data["models"]

    def test_validate_metadata_file_extra_keys(self):
        """Test that a metadata file with extra keys is flagged."""
        invalid_yaml = """
                        models:
                          test-model:
                            architecture: "CNN"
                            extra_key: "not allowed"
                        """
        with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
            tmp.write(invalid_yaml)
            tmp_path = tmp.name

        errors, data = validate_metadata_file(tmp_path)
        os.remove(tmp_path)
        assert len(errors) > 0
        assert any("extra keys" in err for err in errors)

    def test_validate_metadata_file_invalid_top_level(self):
        """Test that a metadata file with an invalid top-level key is flagged."""
        invalid_yaml = """
                        invalid:
                          test-model:
                            architecture: "CNN"
                        """
        with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
            tmp.write(invalid_yaml)
            tmp_path = tmp.name

        errors, data = validate_metadata_file(tmp_path)
        os.remove(tmp_path)
        assert len(errors) > 0
        assert any("not allowed" in err for err in errors)

    def test_validate_metadata_file_non_dict_top_level(self):
        """Test that a non-dictionary top-level YAML structure is flagged."""
        invalid_yaml = """
                        - item1
                        - item2
                        """
        with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
            tmp.write(invalid_yaml)
            tmp_path = tmp.name

        errors, data = validate_metadata_file(tmp_path)
        os.remove(tmp_path)
        assert len(errors) > 0
        assert "Top-level structure must be a dictionary." in errors

    @patch("subprocess.run")
    @patch("subprocess.check_output")
    def test_create_metadata_pr_success(self, mock_check_output, mock_run):
        """Test that create_metadata_pr succeeds with proper subprocess calls."""
        # configure mocks
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        mock_check_output.return_value = "12345"  # mock PR number response

        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_path = os.path.join(tmp_dir, "metadata.yml")
            with open(metadata_path, "w") as f:
                f.write("models: {}")
            
            # Should not raise SystemExit on success
            create_metadata_pr(tmp_dir, branch_name="test-branch")

            # check that git checkout was called with ANY branch name that starts with test-branch
            mock_run.assert_any_call(
                ["git", "checkout", "-b", ANY],
                check=True,
                stdout=ANY,
                stderr=ANY
            )

    @patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd"))
    def test_create_metadata_pr_failure(self, mock_run):
        """Test that create_metadata_pr calls sys.exit when subprocess.run fails."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_path = os.path.join(tmp_dir, "metadata.yml")
            with open(metadata_path, "w") as f:
                f.write("models: {}")
            
            with pytest.raises(SystemExit):
                create_metadata_pr(tmp_dir, branch_name="test-branch")


def test_v2_metadata_is_validated_but_rejected_by_legacy_endpoint(tmp_path):
    from brainscore_core.submission.endpoints import MetadataEndpoint
    from brainscore_core.metadata import dump
    path = tmp_path / 'metadata.yaml'
    path.write_text(dump({'schema_version': '2.0', 'domain': 'vision',
                          'models': {'example': {'model': {'parameter_count': 100}}}}))
    errors, data = validate_metadata_file(path)
    assert errors == []
    assert data['schema_version'] == '2.0'
    endpoint = MetadataEndpoint.__new__(MetadataEndpoint)
    with patch('brainscore_core.submission.endpoints.create_model_meta_entry') as writer:
        with pytest.raises(ValueError, match='trusted merged-PR publisher'):
            endpoint.process_metadata(str(tmp_path), 'models', 'vision')
        writer.assert_not_called()


@pytest.mark.parametrize("filename", ["metadata.yml", "metadata.yaml"])
@pytest.mark.parametrize("legacy", [None, {}, {"runnable": True}])
def test_existing_v2_metadata_only_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    legacy: dict | None,
) -> None:
    from brainscore_core.metadata import dump
    from brainscore_core.plugin_management import handle_metadata

    entry = {"model": {"architecture": {"family": "vision_transformer"}}}
    if legacy is not None:
        entry["legacy"] = legacy
    document = {
        "schema_version": "2.0",
        "domain": "vision",
        "models": {"test_model": entry},
    }
    path = tmp_path / filename
    original = dump(document)
    path.write_text(original)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "handle_metadata",
            "--plugin-dir",
            str(tmp_path),
            "--plugin-type",
            "models",
            "--db-connection",
        ],
    )
    with (
        patch.object(handle_metadata, "generate_metadata") as generate,
        patch.object(handle_metadata, "create_metadata_pr") as create_pr,
        patch.object(handle_metadata, "load_domain_plugin") as load_plugin,
        patch.object(handle_metadata, "MetadataEndpoint") as endpoint,
    ):
        handle_metadata.main()
    generate.assert_not_called()
    create_pr.assert_not_called()
    load_plugin.assert_not_called()
    endpoint.assert_not_called()
    assert path.read_text() == original
    assert json.loads((tmp_path / "validated_metadata.json").read_text()) == document


def test_invalid_existing_v2_metadata_stops_without_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from brainscore_core.plugin_management import handle_metadata

    (tmp_path / "metadata.yaml").write_text(
        'schema_version: "2.0"\ndomain: vision\n'
        "models: {test_model: {unexpected: true}}\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "handle_metadata",
            "--plugin-dir",
            str(tmp_path),
            "--plugin-type",
            "models",
        ],
    )
    with (
        patch.object(handle_metadata, "generate_metadata") as generate,
        patch.object(handle_metadata, "create_metadata_pr") as create_pr,
        pytest.raises(SystemExit) as error,
    ):
        handle_metadata.main()
    assert error.value.code == 1
    generate.assert_not_called()
    create_pr.assert_not_called()
    assert not (tmp_path / "validated_metadata.json").exists()


def test_missing_metadata_still_generates_and_opens_pr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from brainscore_core.plugin_management import handle_metadata

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "handle_metadata",
            "--plugin-dir",
            str(tmp_path),
            "--plugin-type",
            "models",
        ],
    )
    path = tmp_path / "metadata.yml"

    def generate_metadata(*args: object, **kwargs: object) -> str:
        path.write_text("models: {test_model: {architecture: Transformer}}\n")
        return str(path)

    with (
        patch.object(
            handle_metadata, "generate_metadata", side_effect=generate_metadata
        ) as generate,
        patch.object(
            handle_metadata, "create_metadata_pr", return_value="7"
        ) as create_pr,
    ):
        handle_metadata.main()
    generate.assert_called_once_with(str(tmp_path), "models", domain="vision")
    create_pr.assert_called_once_with(str(tmp_path))
    assert json.loads((tmp_path / "validated_metadata.json").read_text()) == {
        "models": {"test_model": {"architecture": "Transformer"}}
    }


def test_existing_legacy_metadata_still_uses_legacy_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from brainscore_core.plugin_management import handle_metadata

    (tmp_path / "metadata.yml").write_text(
        "models: {test_model: {architecture: Transformer}}\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "handle_metadata",
            "--plugin-dir",
            str(tmp_path),
            "--plugin-type",
            "models",
            "--db-connection",
        ],
    )
    plugin = Mock()
    with (
        patch.object(handle_metadata, "generate_metadata") as generate,
        patch.object(handle_metadata, "create_metadata_pr") as create_pr,
        patch.object(handle_metadata, "load_domain_plugin", return_value=plugin),
        patch.object(handle_metadata, "MetadataEndpoint") as endpoint,
    ):
        handle_metadata.main()
    generate.assert_not_called()
    create_pr.assert_not_called()
    endpoint.return_value.assert_called_once_with(
        plugin_dir=str(tmp_path), plugin_type="models", domain="vision"
    )
