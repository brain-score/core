import os
import tempfile
import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock, MagicMock
from brainscore_core.plugin_management.generate_model_metadata import ModelMetadataGenerator
from brainscore_core.plugin_management.domain_plugin_interface import DomainPluginInterface
import importlib.util


def get_installed_package_path(subpath=None):
    """Helper function to get the path to a subdirectory within the installed brainscore_vision package."""
    spec = importlib.util.find_spec('brainscore_vision')
    if spec is None or spec.origin is None:
        raise ImportError("brainscore_vision package not found. Please ensure it is installed.")
        
    package_root = Path(spec.origin).parent.parent
    if subpath:
        full_path = package_root / "brainscore_vision" / subpath
    else:
        full_path = package_root / "brainscore_vision"
    
    if not full_path.exists():
        print(f"Warning: Directory does not exist: {full_path}")
    
    return full_path


class MockModelDomainPlugin(DomainPluginInterface):
    """Mock domain plugin specifically for model testing."""
    
    def __init__(self, benchmark_type="neural"):
        self.benchmark_type = benchmark_type
    
    def load_model(self, identifier: str):
        """Mock model loading."""
        mock_model = Mock()
        mock_model.activations_model = Mock()
        mock_model.activations_model._model = Mock()
        
        # Mock PyTorch model structure
        torch_model = Mock()
        torch_model.parameters = Mock(return_value=[
            Mock(numel=Mock(return_value=1000), requires_grad=True, dim=Mock(return_value=2), element_size=Mock(return_value=4)),
            Mock(numel=Mock(return_value=500), requires_grad=True, dim=Mock(return_value=1), element_size=Mock(return_value=4)),
        ])
        torch_model.modules = Mock(return_value=[torch_model])
        mock_model.activations_model._model = torch_model
        
        return mock_model

    def find_registered_models(self, root_folder: str):
        """Mock model discovery."""
        return ["test_model"]

    def extract_model_for_analysis(self, model):
        """Mock model extraction."""
        return model.activations_model._model

    def detect_model_architecture(self, model, model_name: str):
        """Mock architecture detection."""
        return "DCNN"

    def get_model_family(self, model_name: str):
        """Mock model family detection."""
        if "resnet" in model_name.lower():
            return "resnet"
        return "unknown"

    def create_model_metadata(self, model, model_name: str, model_dir_name: str):
        """Mock model metadata creation."""
        return {
            "architecture": "DCNN",
            "model_family": "resnet",
            "total_parameter_count": 1500,
            "trainable_parameter_count": 1500,
            "total_layers": 1,
            "trainable_layers": 1,
            "model_size_mb": 0.006,
            "training_dataset": None,
            "task_specialization": None,
            "brainscore_link": f"https://github.com/brain-score/vision/tree/master/brainscore_vision/models/{model_dir_name}",
            "huggingface_link": None,
            "extra_notes": None,
            "runnable": False
        }

    # Benchmark methods (minimal implementation for interface compliance)
    def load_benchmark(self, identifier: str): return None
    def create_stimuli_metadata(self, plugin, plugin_dir_name: str): return {}
    def create_data_metadata(self, benchmark, benchmark_dir_name: str): return {}
    def create_metric_metadata(self, plugin, plugin_dir_name: str): return {}
    def find_registered_benchmarks(self, root_folder: str): return []


# =============================================================================
# MODEL METADATA GENERATION - FILE OPERATIONS TESTS
# =============================================================================

class TestModelMetadataFileOperations:
    """Tests focused on model metadata file creation and structure."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = self.temp_dir
        self.model_name = "test_model"
        self.mock_plugin = MockModelDomainPlugin()
        self.generator = ModelMetadataGenerator(self.model_path, self.mock_plugin)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_yaml_file_creation(self):
        """Test that YAML file is created in the correct location."""
        model_list = self.generator.find_registered_models(self.model_path)
        yaml_paths = self.generator(model_list)
        
        yaml_path = yaml_paths[0] if yaml_paths else None
        expected_path = Path(self.model_path) / "metadata.yml"
        
        assert Path(yaml_path).exists()
        assert Path(yaml_path) == expected_path

    def test_yaml_structure_validity(self):
        """Test that generated YAML has correct top-level structure."""
        model_list = self.generator.find_registered_models(self.model_path)
        self.generator(model_list)
        
        yaml_path = Path(self.model_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        assert 'models' in metadata
        assert self.model_name in metadata['models']
        assert isinstance(metadata['models'][self.model_name], dict)

    def test_required_fields_present(self):
        """Test that all required model metadata fields are present."""
        model_list = self.generator.find_registered_models(self.model_path)
        self.generator(model_list)
        
        yaml_path = Path(self.model_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        required_fields = [
            "architecture", "model_family", "total_parameter_count", 
            "trainable_parameter_count", "total_layers", "trainable_layers",
            "model_size_mb", "training_dataset", "task_specialization",
            "brainscore_link", "huggingface_link", "extra_notes", "runnable"
        ]
        
        model_meta = metadata['models'][self.model_name]
        for field in required_fields:
            assert field in model_meta, f"Required field '{field}' missing from model metadata"


# =============================================================================
# MODEL METADATA GENERATION - DATA EXTRACTION TESTS
# =============================================================================

class TestModelMetadataDataExtraction:
    """Tests focused on extracting metadata values from models."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = self.temp_dir
        self.model_name = "test_model"
        self.mock_plugin = MockModelDomainPlugin()
        self.generator = ModelMetadataGenerator(self.model_path, self.mock_plugin)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_architecture_detection(self):
        """Test model architecture is correctly detected."""
        model_list = self.generator.find_registered_models(self.model_path)
        self.generator(model_list)
        
        yaml_path = Path(self.model_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        assert metadata['models'][self.model_name]["architecture"] == "DCNN"

    def test_parameter_count_extraction(self):
        """Test parameter counts are correctly extracted."""
        model_list = self.generator.find_registered_models(self.model_path)
        self.generator(model_list)
        
        yaml_path = Path(self.model_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        model_meta = metadata['models'][self.model_name]
        assert model_meta["total_parameter_count"] == 1500
        assert model_meta["trainable_parameter_count"] == 1500

    def test_model_family_detection(self):
        """Test model family is correctly detected."""
        model_list = self.generator.find_registered_models(self.model_path)
        self.generator(model_list)
        
        yaml_path = Path(self.model_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        assert metadata['models'][self.model_name]["model_family"] == "resnet"

    def test_brainscore_link_generation(self):
        """Test BrainScore links are correctly generated."""
        model_list = self.generator.find_registered_models(self.model_path)
        self.generator(model_list)
        
        yaml_path = Path(self.model_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        expected_link = f"https://github.com/brain-score/vision/tree/master/brainscore_vision/models/{os.path.basename(self.model_path)}"
        assert metadata['models'][self.model_name]["brainscore_link"] == expected_link


# =============================================================================
# MODEL METADATA GENERATION - INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
class TestModelMetadataIntegration:
    """Integration tests with real vision models."""
    
    def setup_method(self):
        try:
            from brainscore_vision.plugin_management import VisionDomainPlugin
            
            self.model_path = str(get_installed_package_path("models/resnet50_tutorial"))
            self.model_name = "resnet50_tutorial"
            self.vision_plugin = VisionDomainPlugin()
            self.generator = ModelMetadataGenerator(self.model_path, self.vision_plugin)
        except ImportError:
            pytest.skip("VisionDomainPlugin not available for integration test")

    def teardown_method(self):
        yaml_path = Path(self.model_path) / "metadata.yml"
        if yaml_path.exists():
            yaml_path.unlink()
            print(f"Deleted: {yaml_path}")

    def test_real_model_discovery(self):
        """Test discovery of real vision models."""
        model_list = self.generator.find_registered_models(self.model_path)
        assert len(model_list) == 1
        assert self.model_name in model_list

    def test_real_model_metadata_extraction(self):
        """Test metadata extraction from real vision models."""
        model_list = self.generator.find_registered_models(self.model_path)
        yaml_paths = self.generator(model_list)
        
        yaml_path = Path(self.model_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        model_meta = metadata['models'][self.model_name]
        
        # Verify real ResNet50 values
        assert model_meta["architecture"] == "DCNN"
        assert model_meta['model_family'] == "resnet"
        assert model_meta["total_parameter_count"] == 25557032
        assert model_meta["trainable_parameter_count"] == 25557032
        assert model_meta["total_layers"] == 151
        assert model_meta["trainable_layers"] == 54
        assert model_meta["model_size_mb"] == 102.23


if __name__ == "__main__":
    # Run unit tests only by default
    pytest.main([__file__, "-v", "-m", "not integration"]) 