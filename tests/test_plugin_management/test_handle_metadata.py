import os
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
