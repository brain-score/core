import os
import tempfile
import pytest
import yaml
from pathlib import Path
from brainscore_core.plugin_management.generate_model_metadata import ModelMetadataGenerator
from brainscore_core.plugin_management.generate_benchmark_metadata import BenchmarkMetadataGenerator
import importlib.util


def get_installed_package_path(subpath=None):
    """
    Helper function to get the path to a subdirectory within the installed brainscore_vision package.
    Addresses issue of different vision paths in different environments.
    
    Args:
        subpath (str, optional): Additional path components to append to the package root.
            For example: "models/resnet50_tutorial" or "benchmarks/rajalingham2020"
    
    Returns:
        Path: The full path to the requested subdirectory
    
    Raises:
        ImportError: If brainscore_vision package is not found
    """
    # Find the actual location of brainscore_vision package
    spec = importlib.util.find_spec('brainscore_vision')
    if spec is None or spec.origin is None:
        raise ImportError("brainscore_vision package not found. Please ensure it is installed.")
        
    # Get the package root directory (parent of __init__.py)
    package_root = Path(spec.origin).parent.parent

    # Construct the full path
    if subpath:
        full_path = package_root / "brainscore_vision" / subpath
    else:
        full_path = package_root / "brainscore_vision"
    
    # Check if the directory exists
    if not full_path.exists():
        print(f"Warning: Directory does not exist: {full_path}")
        parent_dir = full_path.parent
        if parent_dir.exists():
            print("Available directories:")
            print("\n".join(f"  - {d.name}" for d in parent_dir.iterdir() if d.is_dir()))
        else:
            print(f"Parent directory does not exist at: {parent_dir}")
    
    return full_path


class TestModelMetadataGenerator:
    def setup_method(self):
        self.model_path = str(get_installed_package_path("models/resnet50_tutorial"))
        self.model_name = "resnet50_tutorial"
        self.generator = ModelMetadataGenerator(self.model_path)
        self.model_list = self.generator.find_registered_models(self.model_path)
        self.yaml_path_created = self.generator(self.model_list)  # actually generate the YAML here
        self.yaml_path_expected = Path(self.model_path) / "metadata.yml"

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
        assert "model_size_mb" in metadata['models'][self.model_name]
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
        assert metadata['models'][self.model_name]["model_size_mb"] == 102.23
        assert metadata['models'][self.model_name]["training_dataset"] is None
        assert metadata['models'][self.model_name]["task_specialization"] is None
        assert (metadata['models'][self.model_name]["brainscore_link"] ==
                "https://github.com/brain-score/vision/tree/master/brainscore_vision/models/resnet50_tutorial")
        assert metadata['models'][self.model_name]["huggingface_link"] is None
        assert metadata['models'][self.model_name]["extra_notes"] is None


class TestBenchmarkMetadataGeneratorNeural:
    def setup_method(self):
        self.benchmark_path = str(get_installed_package_path("benchmarks/rajalingham2020"))
        self.benchmark_name = "Rajalingham2020.IT-pls"
        self.generator = BenchmarkMetadataGenerator(self.benchmark_path, benchmark_type="neural")
        self.benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        self.yaml_path_created = self.generator(self.benchmark_list)  # actually generate the YAML here
        self.yaml_path_expected = Path(self.benchmark_path) / "metadata.yml"

    def teardown_method(self):
        if self.yaml_path_expected.exists():
            self.yaml_path_expected.unlink()  # Deletes the created YAML file
            print(f"Deleted: {self.yaml_path_expected}")
        else:
            print(f"No metadata.yml file found at: {self.yaml_path_expected}")

    def test_find_registered_benchmarks(self):
        assert len(self.benchmark_list) == 1
        assert self.benchmark_list[0] == self.benchmark_name

    def test_create_yaml_creates_file(self):
        self.yaml_path_created = self.yaml_path_created[0] if self.yaml_path_created else None
        assert Path(self.yaml_path_created).exists()
        assert Path(self.yaml_path_created) == self.yaml_path_expected

    def test_create_yaml_structure(self):
        with open(self.yaml_path_expected, 'r') as f:
            metadata = yaml.safe_load(f)
        assert 'benchmarks' in metadata
        assert self.benchmark_name in metadata['benchmarks']

        # stimulus set fields:
        assert "stimulus_set" in metadata['benchmarks'][self.benchmark_name]
        assert "num_stimuli" in metadata['benchmarks'][self.benchmark_name]['stimulus_set']
        assert "datatype" in metadata['benchmarks'][self.benchmark_name]['stimulus_set']
        assert "stimuli_subtype" in metadata['benchmarks'][self.benchmark_name]['stimulus_set']
        assert "total_size_mb" in metadata['benchmarks'][self.benchmark_name]['stimulus_set']
        assert "brainscore_link" in metadata['benchmarks'][self.benchmark_name]['stimulus_set']
        assert "extra_notes" in metadata['benchmarks'][self.benchmark_name]['stimulus_set']

        # data fields:
        assert "data" in metadata['benchmarks'][self.benchmark_name]
        assert "benchmark_type" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "task" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "region" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "hemisphere" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "num_recording_sites" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "duration_ms" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "species" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "datatype" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "num_subjects" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "pre_processing" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "brainscore_link" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "extra_notes" in metadata['benchmarks'][self.benchmark_name]['data']

        # metric fields
        assert "metric" in metadata['benchmarks'][self.benchmark_name]
        assert "type" in metadata['benchmarks'][self.benchmark_name]['metric']
        assert "reference" in metadata['benchmarks'][self.benchmark_name]['metric']
        assert "public" in metadata['benchmarks'][self.benchmark_name]['metric']
        assert "brainscore_link" in metadata['benchmarks'][self.benchmark_name]['metric']
        assert "extra_notes" in metadata['benchmarks'][self.benchmark_name]['metric']

    def test_benchmark_metadata_methods(self):
        with open(self.yaml_path_expected, 'r') as f:
            metadata = yaml.safe_load(f)
        assert 'benchmarks' in metadata
        assert self.benchmark_name in metadata['benchmarks']

        # stimulus set fields:
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["num_stimuli"] == 616
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["datatype"] == "image"
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["stimuli_subtype"] is None
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["total_size_mb"] == 3.5858
        assert (metadata['benchmarks'][self.benchmark_name]['stimulus_set']["brainscore_link"] ==
                "https://github.com/brain-score/vision/tree/master/brainscore_vision/data/rajalingham2020")
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["extra_notes"] is None

        # data fields:
        assert metadata['benchmarks'][self.benchmark_name]['data']["benchmark_type"] == "neural"
        assert metadata['benchmarks'][self.benchmark_name]['data']["task"] is None
        assert metadata['benchmarks'][self.benchmark_name]['data']["region"] == "IT"
        assert metadata['benchmarks'][self.benchmark_name]['data']["hemisphere"] == "L"
        assert metadata['benchmarks'][self.benchmark_name]['data']["num_recording_sites"] is None
        assert metadata['benchmarks'][self.benchmark_name]['data']["duration_ms"] is None
        assert metadata['benchmarks'][self.benchmark_name]['data']["species"] is None
        assert metadata['benchmarks'][self.benchmark_name]['data']["datatype"] is None
        assert metadata['benchmarks'][self.benchmark_name]['data']["num_subjects"] is None
        assert metadata['benchmarks'][self.benchmark_name]['data']["pre_processing"] is None
        assert (metadata['benchmarks'][self.benchmark_name]['data']["brainscore_link"] ==
                "https://github.com/brain-score/vision/tree/master/brainscore_vision/data/rajalingham2020")
        assert metadata['benchmarks'][self.benchmark_name]['data']["extra_notes"] is None

        # metric fields -> different then production yaml file, as this is generic
        assert metadata['benchmarks'][self.benchmark_name]['metric']["type"] is None
        assert metadata['benchmarks'][self.benchmark_name]['metric']["reference"] is None
        assert metadata['benchmarks'][self.benchmark_name]['metric']["public"] is None
        assert (metadata['benchmarks'][self.benchmark_name]['metric']["brainscore_link"] ==
                "https://github.com/brain-score/vision/tree/master/brainscore_vision/benchmarks/rajalingham2020")
        assert metadata['benchmarks'][self.benchmark_name]['metric']["extra_notes"] is None

class TestBenchmarkMetadataGeneratorBehavioral:
    def setup_method(self):
        self.benchmark_path = str(get_installed_package_path("benchmarks/coggan2024_behavior"))
        self.benchmark_name = "tong.Coggan2024_behavior-ConditionWiseAccuracySimilarity"
        self.generator = BenchmarkMetadataGenerator(self.benchmark_path, benchmark_type="behavioral")
        self.benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        self.yaml_path_created = self.generator(self.benchmark_list)  # actually generate the YAML here
        self.yaml_path_expected = Path(self.benchmark_path) / "metadata.yml"

    def teardown_method(self):
        if self.yaml_path_expected.exists():
            self.yaml_path_expected.unlink()  # Deletes the created YAML file
            print(f"Deleted: {self.yaml_path_expected}")
        else:
            print(f"No metadata.yml file found at: {self.yaml_path_expected}")

    def test_find_registered_benchmarks(self):
        assert len(self.benchmark_list) == 1
        assert self.benchmark_list[0] == self.benchmark_name

    def test_create_yaml_creates_file(self):
        self.yaml_path_created = self.yaml_path_created[0] if self.yaml_path_created else None
        assert Path(self.yaml_path_created).exists()
        assert Path(self.yaml_path_created) == self.yaml_path_expected

    def test_create_yaml_structure(self):
        with open(self.yaml_path_expected, 'r') as f:
            metadata = yaml.safe_load(f)
        assert 'benchmarks' in metadata
        assert self.benchmark_name in metadata['benchmarks']

        # stimulus set fields:
        assert "stimulus_set" in metadata['benchmarks'][self.benchmark_name]
        assert "num_stimuli" in metadata['benchmarks'][self.benchmark_name]['stimulus_set']
        assert "datatype" in metadata['benchmarks'][self.benchmark_name]['stimulus_set']
        assert "stimuli_subtype" in metadata['benchmarks'][self.benchmark_name]['stimulus_set']
        assert "total_size_mb" in metadata['benchmarks'][self.benchmark_name]['stimulus_set']
        assert "brainscore_link" in metadata['benchmarks'][self.benchmark_name]['stimulus_set']
        assert "extra_notes" in metadata['benchmarks'][self.benchmark_name]['stimulus_set']

        # data fields:
        assert "data" in metadata['benchmarks'][self.benchmark_name]
        assert "benchmark_type" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "task" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "region" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "hemisphere" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "num_recording_sites" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "duration_ms" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "species" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "datatype" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "num_subjects" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "pre_processing" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "brainscore_link" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "extra_notes" in metadata['benchmarks'][self.benchmark_name]['data']

        # metric fields
        assert "metric" in metadata['benchmarks'][self.benchmark_name]
        assert "type" in metadata['benchmarks'][self.benchmark_name]['metric']
        assert "reference" in metadata['benchmarks'][self.benchmark_name]['metric']
        assert "public" in metadata['benchmarks'][self.benchmark_name]['metric']
        assert "brainscore_link" in metadata['benchmarks'][self.benchmark_name]['metric']
        assert "extra_notes" in metadata['benchmarks'][self.benchmark_name]['metric']

    def test_benchmark_metadata_methods(self):
        with open(self.yaml_path_expected, 'r') as f:
            metadata = yaml.safe_load(f)
        assert 'benchmarks' in metadata
        assert self.benchmark_name in metadata['benchmarks']

        # stimulus set fields:
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["num_stimuli"] == 22560
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["datatype"] == "image"
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["stimuli_subtype"] is None
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["total_size_mb"] == 12.6584
        assert (metadata['benchmarks'][self.benchmark_name]['stimulus_set']["brainscore_link"] ==
                "https://github.com/brain-score/vision/tree/master/brainscore_vision/data/coggan2024_behavior")
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["extra_notes"] is None

        # data fields:
        assert metadata['benchmarks'][self.benchmark_name]['data']["benchmark_type"] == "behavioral"
        assert metadata['benchmarks'][self.benchmark_name]['data']["task"] is None
        assert metadata['benchmarks'][self.benchmark_name]['data']["region"] is None
        assert metadata['benchmarks'][self.benchmark_name]['data']["hemisphere"] is None
        assert metadata['benchmarks'][self.benchmark_name]['data']["num_recording_sites"] is None
        assert metadata['benchmarks'][self.benchmark_name]['data']["duration_ms"] is None
        assert metadata['benchmarks'][self.benchmark_name]['data']["species"] is None
        assert metadata['benchmarks'][self.benchmark_name]['data']["datatype"] == "behavioral"
        assert metadata['benchmarks'][self.benchmark_name]['data']["num_subjects"] == 30
        assert metadata['benchmarks'][self.benchmark_name]['data']["pre_processing"] is None
        assert (metadata['benchmarks'][self.benchmark_name]['data']["brainscore_link"] ==
                "https://github.com/brain-score/vision/tree/master/brainscore_vision/data/coggan2024_behavior")
        assert metadata['benchmarks'][self.benchmark_name]['data']["extra_notes"] is None

        # metric fields -> different then production yaml file, as this is generic
        assert metadata['benchmarks'][self.benchmark_name]['metric']["type"] is None
        assert metadata['benchmarks'][self.benchmark_name]['metric']["reference"] is None
        assert metadata['benchmarks'][self.benchmark_name]['metric']["public"] is None
        assert (metadata['benchmarks'][self.benchmark_name]['metric']["brainscore_link"] ==
                "https://github.com/brain-score/vision/tree/master/brainscore_vision/benchmarks/coggan2024_behavior")
        assert metadata['benchmarks'][self.benchmark_name]['metric']["extra_notes"] is None

class TestBenchmarkMetadataGeneratorEngineering:
    def setup_method(self):
        self.benchmark_path = str(get_installed_package_path("benchmarks/objectnet"))
        self.benchmark_name = "ObjectNet-top1"
        self.generator = BenchmarkMetadataGenerator(self.benchmark_path, benchmark_type="engineering")
        self.benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        self.yaml_path_created = self.generator(self.benchmark_list)  # actually generate the YAML here
        self.yaml_path_expected = Path(self.benchmark_path) / "metadata.yml"

    def teardown_method(self):
        if self.yaml_path_expected.exists():
            self.yaml_path_expected.unlink()  # Deletes the created YAML file
            print(f"Deleted: {self.yaml_path_expected}")
        else:
            print(f"No metadata.yml file found at: {self.yaml_path_expected}")

    def test_find_registered_benchmarks(self):
        assert len(self.benchmark_list) == 1
        assert self.benchmark_list[0] == self.benchmark_name

    def test_create_yaml_creates_file(self):
        self.yaml_path_created = self.yaml_path_created[0] if self.yaml_path_created else None
        assert Path(self.yaml_path_created).exists()
        assert Path(self.yaml_path_created) == self.yaml_path_expected

    def test_create_yaml_structure(self):
        with open(self.yaml_path_expected, 'r') as f:
            metadata = yaml.safe_load(f)
        assert 'benchmarks' in metadata
        assert self.benchmark_name in metadata['benchmarks']

        # stimulus set fields:
        assert "stimulus_set" in metadata['benchmarks'][self.benchmark_name]
        assert "num_stimuli" in metadata['benchmarks'][self.benchmark_name]['stimulus_set']
        assert "datatype" in metadata['benchmarks'][self.benchmark_name]['stimulus_set']
        assert "stimuli_subtype" in metadata['benchmarks'][self.benchmark_name]['stimulus_set']
        assert "total_size_mb" in metadata['benchmarks'][self.benchmark_name]['stimulus_set']
        assert "brainscore_link" in metadata['benchmarks'][self.benchmark_name]['stimulus_set']
        assert "extra_notes" in metadata['benchmarks'][self.benchmark_name]['stimulus_set']

        # data fields:
        assert "data" in metadata['benchmarks'][self.benchmark_name]
        assert "benchmark_type" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "task" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "region" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "hemisphere" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "num_recording_sites" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "duration_ms" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "species" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "datatype" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "num_subjects" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "pre_processing" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "brainscore_link" in metadata['benchmarks'][self.benchmark_name]['data']
        assert "extra_notes" in metadata['benchmarks'][self.benchmark_name]['data']

        # metric fields
        assert "metric" in metadata['benchmarks'][self.benchmark_name]
        assert "type" in metadata['benchmarks'][self.benchmark_name]['metric']
        assert "reference" in metadata['benchmarks'][self.benchmark_name]['metric']
        assert "public" in metadata['benchmarks'][self.benchmark_name]['metric']
        assert "brainscore_link" in metadata['benchmarks'][self.benchmark_name]['metric']
        assert "extra_notes" in metadata['benchmarks'][self.benchmark_name]['metric']

    def test_benchmark_metadata_methods(self):
        with open(self.yaml_path_expected, 'r') as f:
            metadata = yaml.safe_load(f)
        assert 'benchmarks' in metadata
        assert self.benchmark_name in metadata['benchmarks']

        # stimulus set fields:
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["num_stimuli"] is None
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["datatype"] == "image"
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["stimuli_subtype"] is None
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["total_size_mb"] is None
        assert (metadata['benchmarks'][self.benchmark_name]['stimulus_set']["brainscore_link"] ==
                "https://github.com/brain-score/vision/tree/master/brainscore_vision/data/objectnet")
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["extra_notes"] is None

        # data fields:
        assert metadata['benchmarks'][self.benchmark_name]['data']["benchmark_type"] == "engineering"
        assert metadata['benchmarks'][self.benchmark_name]['data']["task"] is None
        assert metadata['benchmarks'][self.benchmark_name]['data']["region"] is None
        assert metadata['benchmarks'][self.benchmark_name]['data']["hemisphere"] is None
        assert metadata['benchmarks'][self.benchmark_name]['data']["num_recording_sites"] is None
        assert metadata['benchmarks'][self.benchmark_name]['data']["duration_ms"] is None
        assert metadata['benchmarks'][self.benchmark_name]['data']["species"] is None
        assert metadata['benchmarks'][self.benchmark_name]['data']["datatype"] == "engineering"
        assert metadata['benchmarks'][self.benchmark_name]['data']["num_subjects"] is None
        assert metadata['benchmarks'][self.benchmark_name]['data']["pre_processing"] is None
        assert (metadata['benchmarks'][self.benchmark_name]['data']["brainscore_link"] ==
                "https://github.com/brain-score/vision/tree/master/brainscore_vision/data/objectnet")
        assert metadata['benchmarks'][self.benchmark_name]['data']["extra_notes"] is None

        # metric fields -> different then production yaml file, as this is generic
        assert metadata['benchmarks'][self.benchmark_name]['metric']["type"] is None
        assert metadata['benchmarks'][self.benchmark_name]['metric']["reference"] is None
        assert metadata['benchmarks'][self.benchmark_name]['metric']["public"] is None
        assert (metadata['benchmarks'][self.benchmark_name]['metric']["brainscore_link"] ==
                "https://github.com/brain-score/vision/tree/master/brainscore_vision/benchmarks/objectnet")
        assert metadata['benchmarks'][self.benchmark_name]['metric']["extra_notes"] is None