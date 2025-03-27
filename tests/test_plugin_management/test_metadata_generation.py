import os
import tempfile
import pytest
import yaml
from pathlib import Path
from brainscore_core.plugin_management.generate_model_metadata import ModelMetadataGenerator


class TestModelMetadataGenerator:
    def setup_method(self):
        self.model_path = "vision/brainscore_vision/models/resnet50_tutorial"
        self.model_name = "resnet50_tutorial"
        self.generator = ModelMetadataGenerator(self.model_path)
        self.model_list = self.generator.find_registered_models(self.model_path)
        self.yaml_path_created = self.generator(self.model_list)  # actually generate the YAML here
        self.yaml_path_expected = Path(self.model_path) / "metadata.yaml"

    def teardown_method(self):
        if self.yaml_path_expected.exists():
            self.yaml_path_expected.unlink()  # Deletes the created YAML file
            print(f"Deleted: {self.yaml_path_expected}")
        else:
            print(f"No metadata.yml file found at: {self.yaml_path_expected}")


    def test_find_registered_models(self):
        assert len(self.model_list) == 1
        assert self.model_list[0] == self.model_name

    def test_create_yaml_creates_file(self):
        self.yaml_path_created = self.yaml_path_created[0] if self.yaml_path_created else None
        assert Path(self.yaml_path_created).exists()
        assert Path(self.yaml_path_created) == self.yaml_path_expected

    def test_create_yaml_structure(self):
        with open(self.yaml_path_expected, 'r') as f:
            metadata = yaml.safe_load(f)
        assert 'models' in metadata
        assert self.model_name in metadata['models']
        assert "architecture" in metadata['models'][self.model_name]
        assert "model_family" in metadata['models'][self.model_name]
        assert "total_parameter_count" in metadata['models'][self.model_name]
        assert "trainable_parameter_count" in metadata['models'][self.model_name]
        assert "total_layers" in metadata['models'][self.model_name]
        assert "trainable_layers" in metadata['models'][self.model_name]
        assert "model_size_MB" in metadata['models'][self.model_name]
        assert "training_dataset" in metadata['models'][self.model_name]
        assert "task_specialization" in metadata['models'][self.model_name]
        assert "brainscore_link" in metadata['models'][self.model_name]
        assert "huggingface_link" in metadata['models'][self.model_name]
        assert "extra_notes" in metadata['models'][self.model_name]

    def test_model_metadata_methods(self):
        with open(self.yaml_path_expected, 'r') as f:
            metadata = yaml.safe_load(f)
        assert 'models' in metadata
        assert self.model_name in metadata['models']
        assert metadata['models'][self.model_name]["architecture"] == "DCNN"
        assert metadata['models'][self.model_name]['model_family'] == "resnet"
        assert metadata['models'][self.model_name]["total_parameter_count"] == 25557032
        assert metadata['models'][self.model_name]["trainable_parameter_count"] == 25557032
        assert metadata['models'][self.model_name]["total_layers"] == 151
        assert metadata['models'][self.model_name]["trainable_layers"] == 54
        assert metadata['models'][self.model_name]["model_size_MB"] == 102.23
        assert metadata['models'][self.model_name]["training_dataset"] is None
        assert metadata['models'][self.model_name]["task_specialization"] is None
        assert (metadata['models'][self.model_name]["brainscore_link"] ==
                "https://github.com/brain-score/vision/tree/master/brainscore_vision/models/resnet50_tutorial")
        assert metadata['models'][self.model_name]["huggingface_link"] is None
        assert metadata['models'][self.model_name]["extra_notes"] is None

class TestBenchmarkMetadataGenerator:
    def setup_method(self):
        self.model_path = "vision/brainscore_vision/models/resnet50_tutorial"
        self.model_name = "resnet50_tutorial"
        self.generator = ModelMetadataGenerator(self.model_path)
        self.model_list = self.generator.find_registered_models(self.model_path)
        self.yaml_path_created = self.generator(self.model_list)  # actually generate the YAML here
        self.yaml_path_expected = Path(self.model_path) / "metadata.yaml"

    def teardown_method(self):
        if self.yaml_path_expected.exists():
            self.yaml_path_expected.unlink()  # Deletes the created YAML file
            print(f"Deleted: {self.yaml_path_expected}")
        else:
            print(f"No metadata.yml file found at: {self.yaml_path_expected}")


    def test_find_registered_models(self):
        assert len(self.model_list) == 1
        assert self.model_list[0] == self.model_name

    def test_create_yaml_creates_file(self):
        self.yaml_path_created = self.yaml_path_created[0] if self.yaml_path_created else None
        assert Path(self.yaml_path_created).exists()
        assert Path(self.yaml_path_created) == self.yaml_path_expected

    def test_create_yaml_structure(self):
        with open(self.yaml_path_expected, 'r') as f:
            metadata = yaml.safe_load(f)
        assert 'models' in metadata
        assert self.model_name in metadata['models']
        assert "architecture" in metadata['models'][self.model_name]
        assert "model_family" in metadata['models'][self.model_name]
        assert "total_parameter_count" in metadata['models'][self.model_name]
        assert "trainable_parameter_count" in metadata['models'][self.model_name]
        assert "total_layers" in metadata['models'][self.model_name]
        assert "trainable_layers" in metadata['models'][self.model_name]
        assert "model_size_MB" in metadata['models'][self.model_name]
        assert "training_dataset" in metadata['models'][self.model_name]
        assert "task_specialization" in metadata['models'][self.model_name]
        assert "brainscore_link" in metadata['models'][self.model_name]
        assert "huggingface_link" in metadata['models'][self.model_name]
        assert "extra_notes" in metadata['models'][self.model_name]

    def test_model_metadata_methods(self):
        with open(self.yaml_path_expected, 'r') as f:
            metadata = yaml.safe_load(f)
        assert 'models' in metadata
        assert self.model_name in metadata['models']
        assert metadata['models'][self.model_name]["architecture"] == "DCNN"
        assert metadata['models'][self.model_name]['model_family'] == "resnet"
        assert metadata['models'][self.model_name]["total_parameter_count"] == 25557032
        assert metadata['models'][self.model_name]["trainable_parameter_count"] == 25557032
        assert metadata['models'][self.model_name]["total_layers"] == 151
        assert metadata['models'][self.model_name]["trainable_layers"] == 54
        assert metadata['models'][self.model_name]["model_size_MB"] == 102.23
        assert metadata['models'][self.model_name]["training_dataset"] is None
        assert metadata['models'][self.model_name]["task_specialization"] is None
        assert (metadata['models'][self.model_name]["brainscore_link"] ==
                "https://github.com/brain-score/vision/tree/master/brainscore_vision/models/resnet50_tutorial")
        assert metadata['models'][self.model_name]["huggingface_link"] is None
        assert metadata['models'][self.model_name]["extra_notes"] is None
