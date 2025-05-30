import os
import tempfile
import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock, MagicMock
from brainscore_core.plugin_management.generate_model_metadata import ModelMetadataGenerator
from brainscore_core.plugin_management.generate_benchmark_metadata import BenchmarkMetadataGenerator
from brainscore_core.plugin_management.domain_plugin_interface import DomainPluginInterface
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


class MockDomainPlugin(DomainPluginInterface):
    """Mock domain plugin for fast unit testing."""
    
    def __init__(self, benchmark_type="neural"):
        self.benchmark_type = benchmark_type
        self.mock_benchmark = Mock()
        self.mock_benchmark._assembly = Mock()
        self.mock_benchmark._assembly.stimulus_set = Mock()
        
        # Configure mock for different benchmark types
        self.setup_mock_data()
    
    def setup_mock_data(self):
        """Setup mock data based on benchmark type."""
        # Mock stimulus set
        stimulus_set = self.mock_benchmark._assembly.stimulus_set
        stimulus_set.__len__ = Mock(return_value=616 if self.benchmark_type == "neural" else 100)
        stimulus_set.memory_usage = Mock(return_value=Mock())
        
        if self.benchmark_type == "neural":
            stimulus_set.memory_usage.return_value.sum = Mock(return_value=3758038)  # ~3.6MB
        else:
            stimulus_set.memory_usage.return_value.sum = Mock(return_value=1024*1024*10)  # 10MB
        
        # Mock assembly for data metadata
        assembly = self.mock_benchmark._assembly
        assembly.hemisphere = Mock()
        assembly.hemisphere.values = ["L", "L", "L"]
        assembly.subject = Mock() 
        assembly.subject.values = ["sub1", "sub2", "sub1", "sub2"]
        assembly.region = Mock()
        assembly.region.values = ["IT", "IT", "IT"]
    
    def load_benchmark(self, identifier: str):
        """Mock benchmark loading."""
        # Simulate successful loading for test benchmarks
        if "test" in identifier.lower() or identifier in ["Rajalingham2020.IT-pls", "ObjectNet-top1", "tong.Coggan2024_behavior-ConditionWiseAccuracySimilarity"]:
            return self.mock_benchmark
        return None
    
    def create_stimuli_metadata(self, plugin, plugin_dir_name: str):
        """Mock stimuli metadata creation with realistic values."""
        if self.benchmark_type == "neural":
            return {
                "num_stimuli": 616,
                "datatype": "image",
                "stimuli_subtype": None,
                "total_size_MB": 3.5858,
                "brainscore_link": f"https://github.com/brain-score/vision/tree/master/brainscore_vision/data/{plugin_dir_name}",
                "extra_notes": None
            }
        elif self.benchmark_type == "behavioral":
            return {
                "num_stimuli": 22560,
                "datatype": "image", 
                "stimuli_subtype": None,
                "total_size_MB": 12.6584,
                "brainscore_link": f"https://github.com/brain-score/vision/tree/master/brainscore_vision/data/{plugin_dir_name}",
                "extra_notes": None
            }
        else:  # engineering
            return {
                "num_stimuli": None,
                "datatype": "image",
                "stimuli_subtype": None,
                "total_size_MB": None,
                "brainscore_link": f"https://github.com/brain-score/vision/tree/master/brainscore_vision/data/{plugin_dir_name}",
                "extra_notes": None
            }
    
    def create_data_metadata(self, benchmark, benchmark_dir_name: str):
        """Mock data metadata creation with realistic values."""
        base_metadata = {
            "benchmark_type": self.benchmark_type,
            "task": None,
            "num_recording_sites": None,
            "duration_ms": None,
            "species": None,
            "pre_processing": None,
            "brainscore_link": f"https://github.com/brain-score/vision/tree/master/brainscore_vision/data/{benchmark_dir_name}",
            "extra_notes": None
        }
        
        if self.benchmark_type == "neural":
            base_metadata.update({
                "region": "IT",
                "hemisphere": "L",
                "datatype": None,
                "num_subjects": 2,
            })
        elif self.benchmark_type == "behavioral":
            base_metadata.update({
                "region": None,
                "hemisphere": None,
                "datatype": "behavioral",
                "num_subjects": 30,
            })
        else:  # engineering
            base_metadata.update({
                "region": None,
                "hemisphere": None,
                "datatype": "engineering",
                "num_subjects": None,
            })
        
        return base_metadata
    
    def create_metric_metadata(self, plugin, plugin_dir_name: str):
        """Mock metric metadata creation."""
        return {
            "type": None,
            "reference": None,
            "public": None,
            "brainscore_link": f"https://github.com/brain-score/vision/tree/master/brainscore_vision/benchmarks/{plugin_dir_name}",
            "extra_notes": None
        }
    
    def find_registered_benchmarks(self, root_folder: str):
        """Mock benchmark discovery."""
        # Return realistic benchmark names based on folder
        if "rajalingham2020" in root_folder:
            return ["Rajalingham2020.IT-pls"]
        elif "coggan2024_behavior" in root_folder:
            return ["tong.Coggan2024_behavior-ConditionWiseAccuracySimilarity"]
        elif "objectnet" in root_folder:
            return ["ObjectNet-top1"]
        else:
            return ["test_benchmark"]

    # ============================================================================
    # MODEL-RELATED METHODS
    # ============================================================================
    
    def load_model(self, identifier: str):
        """Mock model loading."""
        # Create a realistic mock model for any identifier
        mock_model = Mock()
        mock_model.activations_model = Mock()
        mock_model.activations_model._model = Mock()
        
        # Mock PyTorch model structure
        torch_model = Mock()
        torch_model.parameters = Mock(return_value=[
            Mock(numel=Mock(return_value=1000), requires_grad=True, dim=Mock(return_value=2), element_size=Mock(return_value=4)),
            Mock(numel=Mock(return_value=500), requires_grad=True, dim=Mock(return_value=1), element_size=Mock(return_value=4)),
        ])
        torch_model.modules = Mock(return_value=[torch_model])  # Simplified
        
        mock_model.activations_model._model = torch_model
        return mock_model
    
    def find_registered_models(self, root_folder: str):
        """Mock model discovery."""
        if "resnet50_tutorial" in root_folder:
            return ["resnet50_tutorial"]
        else:
            return ["test_model"]
    
    def extract_model_for_analysis(self, model):
        """Mock model extraction."""
        try:
            return model.activations_model._model
        except AttributeError:
            return model
    
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
            "model_size_MB": 0.006,
            "training_dataset": None,
            "task_specialization": None,
            "brainscore_link": f"https://github.com/brain-score/vision/tree/master/brainscore_vision/models/{model_dir_name}",
            "huggingface_link": None,
            "extra_notes": None
        }

# =============================================================================
# UNIT TESTS WITH VISION DOMAIN PLUGIN (Requires brainscore_vision package)
# =============================================================================
class TestModelMetadataGenerator:
    def setup_method(self):
        try:
            from brainscore_vision.plugin_management import VisionDomainPlugin
            
            self.model_path = str(get_installed_package_path("models/resnet50_tutorial"))
            self.model_name = "resnet50_tutorial"
            self.vision_plugin = VisionDomainPlugin()
            self.generator = ModelMetadataGenerator(self.model_path, self.vision_plugin)
            self.model_list = self.generator.find_registered_models(self.model_path)
            self.yaml_path_created = self.generator(self.model_list)  # actually generate the YAML here
            self.yaml_path_expected = Path(self.model_path) / "metadata.yml"
        except ImportError:
            pytest.skip("VisionDomainPlugin not available for model tests")

    def teardown_method(self):
        if hasattr(self, 'yaml_path_expected') and self.yaml_path_expected.exists():
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


# =============================================================================
# UNIT TESTS WITH MOCK PLUGINS (Fast, isolated testing)
# =============================================================================

class TestModelMetadataGeneratorMock:
    """Unit tests for ModelMetadataGenerator using mock plugin."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = self.temp_dir
        self.model_name = "test_model"
        self.mock_plugin = MockDomainPlugin()
        self.generator = ModelMetadataGenerator(self.model_path, self.mock_plugin)
        self.model_list = self.generator.find_registered_models(self.model_path)
        self.yaml_path_created = self.generator(self.model_list)
        self.yaml_path_expected = Path(self.model_path) / "metadata.yml"

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_find_registered_models(self):
        assert len(self.model_list) == 1
        assert self.model_list[0] == "test_model"  # Mock returns this

    def test_create_yaml_creates_file(self):
        self.yaml_path_created = self.yaml_path_created[0] if self.yaml_path_created else None
        assert Path(self.yaml_path_created).exists()
        assert Path(self.yaml_path_created) == self.yaml_path_expected

    def test_create_yaml_structure(self):
        with open(self.yaml_path_expected, 'r') as f:
            metadata = yaml.safe_load(f)
        assert 'models' in metadata
        assert "test_model" in metadata['models']

        # Check all required fields exist
        required_fields = [
            "architecture", "model_family", "total_parameter_count", 
            "trainable_parameter_count", "total_layers", "trainable_layers",
            "model_size_MB", "training_dataset", "task_specialization",
            "brainscore_link", "huggingface_link", "extra_notes"
        ]
        for field in required_fields:
            assert field in metadata['models']["test_model"]

    def test_model_metadata_methods(self):
        with open(self.yaml_path_expected, 'r') as f:
            metadata = yaml.safe_load(f)
        assert 'models' in metadata
        assert "test_model" in metadata['models']

        # Verify mock values
        assert metadata['models']["test_model"]["architecture"] == "DCNN"
        assert metadata['models']["test_model"]["model_family"] == "resnet"
        assert metadata['models']["test_model"]["total_parameter_count"] == 1500
        assert metadata['models']["test_model"]["trainable_parameter_count"] == 1500


class TestBenchmarkMetadataGeneratorNeuralMock:
    """Unit tests for neural benchmarks using mock plugin."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.benchmark_path = self.temp_dir
        self.benchmark_name = "test_benchmark"  # Use mock benchmark name
        self.mock_plugin = MockDomainPlugin(benchmark_type="neural")
        self.generator = BenchmarkMetadataGenerator(self.benchmark_path, self.mock_plugin)
        self.benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        self.yaml_path_created = self.generator(self.benchmark_list)
        self.yaml_path_expected = Path(self.benchmark_path) / "metadata.yml"

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generator_initialization(self):
        """Test that the generator properly initializes with a domain plugin."""
        assert self.generator.benchmark_dir == self.benchmark_path
        assert self.generator.domain_plugin == self.mock_plugin

    def test_find_registered_benchmarks_delegates_to_plugin(self):
        """Test that benchmark discovery is delegated to the domain plugin."""
        benchmarks = self.generator.find_registered_benchmarks("/fake/path")
        assert benchmarks == ["test_benchmark"]

    def test_find_registered_benchmarks(self):
        assert len(self.benchmark_list) >= 1, f"Expected at least 1 benchmark, found {len(self.benchmark_list)}: {self.benchmark_list}"
        assert self.benchmark_name in self.benchmark_list, f"Expected benchmark '{self.benchmark_name}' not found in {self.benchmark_list}"

    def test_create_yaml_creates_file(self):
        self.yaml_path_created = self.yaml_path_created[0] if self.yaml_path_created else None
        assert Path(self.yaml_path_created).exists()
        assert Path(self.yaml_path_created) == self.yaml_path_expected

    def test_create_yaml_structure(self):
        with open(self.yaml_path_expected, 'r') as f:
            metadata = yaml.safe_load(f)
        assert 'benchmarks' in metadata
        assert self.benchmark_name in metadata['benchmarks']

        # Verify structure exists (values may vary depending on actual benchmark)
        benchmark_meta = metadata['benchmarks'][self.benchmark_name]
        
        # Check stimulus_set metadata structure
        assert 'stimulus_set' in benchmark_meta
        assert 'datatype' in benchmark_meta['stimulus_set']
        assert benchmark_meta['stimulus_set']['datatype'] == "image"
        
        # Check data metadata structure
        assert 'data' in benchmark_meta
        assert 'benchmark_type' in benchmark_meta['data']
        assert benchmark_meta['data']['benchmark_type'] == "neural"

    def test_benchmark_metadata_methods(self):
        with open(self.yaml_path_expected, 'r') as f:
            metadata = yaml.safe_load(f)
        assert 'benchmarks' in metadata
        assert self.benchmark_name in metadata['benchmarks']

        # Verify structure exists (values may vary depending on actual benchmark)
        benchmark_meta = metadata['benchmarks'][self.benchmark_name]
        
        # Check stimulus_set metadata structure
        assert 'stimulus_set' in benchmark_meta
        assert 'datatype' in benchmark_meta['stimulus_set']
        assert benchmark_meta['stimulus_set']['datatype'] == "image"
        
        # Check data metadata structure
        assert 'data' in benchmark_meta
        assert 'benchmark_type' in benchmark_meta['data']
        assert benchmark_meta['data']['benchmark_type'] == "neural"


class TestBenchmarkMetadataGeneratorBehavioralMock:
    """Unit tests for behavioral benchmarks using mock plugin."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.benchmark_path = self.temp_dir
        self.benchmark_name = "test_benchmark"  # Use mock benchmark name
        self.mock_plugin = MockDomainPlugin(benchmark_type="behavioral")
        self.generator = BenchmarkMetadataGenerator(self.benchmark_path, self.mock_plugin)
        self.benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        self.yaml_path_created = self.generator(self.benchmark_list)
        self.yaml_path_expected = Path(self.benchmark_path) / "metadata.yml"

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_benchmark_metadata_methods(self):
        with open(self.yaml_path_expected, 'r') as f:
            metadata = yaml.safe_load(f)
        assert 'benchmarks' in metadata
        assert self.benchmark_name in metadata['benchmarks']

        # stimulus set fields:
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["num_stimuli"] == 22560
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["datatype"] == "image"
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["total_size_MB"] == 12.6584

        # data fields:
        assert metadata['benchmarks'][self.benchmark_name]['data']["benchmark_type"] == "behavioral"
        assert metadata['benchmarks'][self.benchmark_name]['data']["datatype"] == "behavioral"
        assert metadata['benchmarks'][self.benchmark_name]['data']["num_subjects"] == 30


class TestBenchmarkMetadataGeneratorEngineeringMock:
    """Unit tests for engineering benchmarks using mock plugin."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.benchmark_path = self.temp_dir
        self.benchmark_name = "test_benchmark"  # Use mock benchmark name
        self.mock_plugin = MockDomainPlugin(benchmark_type="engineering")
        self.generator = BenchmarkMetadataGenerator(self.benchmark_path, self.mock_plugin)
        self.benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        self.yaml_path_created = self.generator(self.benchmark_list)
        self.yaml_path_expected = Path(self.benchmark_path) / "metadata.yml"

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_benchmark_metadata_methods(self):
        with open(self.yaml_path_expected, 'r') as f:
            metadata = yaml.safe_load(f)
        assert 'benchmarks' in metadata
        assert self.benchmark_name in metadata['benchmarks']

        # stimulus set fields:
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["num_stimuli"] is None
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["datatype"] == "image"
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["total_size_MB"] is None

        # data fields:
        assert metadata['benchmarks'][self.benchmark_name]['data']["benchmark_type"] == "engineering"
        assert metadata['benchmarks'][self.benchmark_name]['data']["datatype"] == "engineering"
        assert metadata['benchmarks'][self.benchmark_name]['data']["num_subjects"] is None


# =============================================================================
# INTEGRATION TESTS WITH REAL VISION PLUGIN (Comprehensive end-to-end testing)
# =============================================================================

@pytest.mark.integration
class TestBenchmarkMetadataGeneratorNeuralIntegration:
    """Integration tests for neural benchmarks using real VisionDomainPlugin."""
    
    def setup_method(self):
        try:
            from brainscore_vision.plugin_management import VisionDomainPlugin
            
            # Try to find the rajalingham2020 benchmark in local vision directory structure
            import os
            local_benchmark_path = os.path.join(os.getcwd(), "vision", "brainscore_vision", "benchmarks", "rajalingham2020")
            
            if os.path.exists(local_benchmark_path):
                self.benchmark_path = local_benchmark_path
            else:
                # Fallback to the original approach if the local path doesn't exist
                self.benchmark_path = str(get_installed_package_path("benchmarks/rajalingham2020"))
            
            self.benchmark_name = "Rajalingham2020.IT-pls"
            self.vision_plugin = VisionDomainPlugin(benchmark_type="neural")
            self.generator = BenchmarkMetadataGenerator(self.benchmark_path, self.vision_plugin)
            self.benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
            self.yaml_path_created = self.generator(self.benchmark_list)
            self.yaml_path_expected = Path(self.benchmark_path) / "metadata.yml"
        except ImportError:
            pytest.skip("VisionDomainPlugin not available for integration test")

    def teardown_method(self):
        if hasattr(self, 'yaml_path_expected') and self.yaml_path_expected.exists():
            self.yaml_path_expected.unlink()
            print(f"Deleted: {self.yaml_path_expected}")

    def test_find_registered_benchmarks(self):
        assert len(self.benchmark_list) >= 1, f"Expected at least 1 benchmark, found {len(self.benchmark_list)}: {self.benchmark_list}"
        assert self.benchmark_name in self.benchmark_list, f"Expected benchmark '{self.benchmark_name}' not found in {self.benchmark_list}"

    def test_create_yaml_creates_file(self):
        self.yaml_path_created = self.yaml_path_created[0] if self.yaml_path_created else None
        assert Path(self.yaml_path_created).exists()
        assert Path(self.yaml_path_created) == self.yaml_path_expected

    def test_benchmark_metadata_methods(self):
        with open(self.yaml_path_expected, 'r') as f:
            metadata = yaml.safe_load(f)
        assert 'benchmarks' in metadata
        assert self.benchmark_name in metadata['benchmarks']

        # Verify real data values for Rajalingham2020
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["num_stimuli"] == 616
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["datatype"] == "image"
        assert metadata['benchmarks'][self.benchmark_name]['data']["benchmark_type"] == "neural"
        assert metadata['benchmarks'][self.benchmark_name]['data']["region"] == "IT"
        assert metadata['benchmarks'][self.benchmark_name]['data']["hemisphere"] == "L"


@pytest.mark.integration 
class TestBenchmarkMetadataGeneratorBehavioralIntegration:
    """Integration tests for behavioral benchmarks using real VisionDomainPlugin."""
    
    def setup_method(self):
        try:
            from brainscore_vision.plugin_management import VisionDomainPlugin
            
            # Try to find the coggan2024_behavior benchmark in local vision directory structure
            import os
            local_benchmark_path = os.path.join(os.getcwd(), "vision", "brainscore_vision", "benchmarks", "coggan2024_behavior")
            
            if os.path.exists(local_benchmark_path):
                self.benchmark_path = local_benchmark_path
            else:
                # Fallback to the original approach if the local path doesn't exist
                self.benchmark_path = str(get_installed_package_path("benchmarks/coggan2024_behavior"))
            
            self.benchmark_name = "tong.Coggan2024_behavior-ConditionWiseAccuracySimilarity"
            self.vision_plugin = VisionDomainPlugin(benchmark_type="behavioral")
            self.generator = BenchmarkMetadataGenerator(self.benchmark_path, self.vision_plugin)
            self.benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
            self.yaml_path_created = self.generator(self.benchmark_list)
            self.yaml_path_expected = Path(self.benchmark_path) / "metadata.yml"
        except ImportError:
            pytest.skip("VisionDomainPlugin not available for integration test")

    def teardown_method(self):
        if hasattr(self, 'yaml_path_expected') and self.yaml_path_expected.exists():
            self.yaml_path_expected.unlink()
            print(f"Deleted: {self.yaml_path_expected}")

    def test_benchmark_metadata_methods(self):
        with open(self.yaml_path_expected, 'r') as f:
            metadata = yaml.safe_load(f)
        assert 'benchmarks' in metadata
        assert self.benchmark_name in metadata['benchmarks']

        # Verify real data values for Coggan2024
        assert metadata['benchmarks'][self.benchmark_name]['data']["benchmark_type"] == "behavioral"
        assert metadata['benchmarks'][self.benchmark_name]['data']["datatype"] == "behavioral"
        assert metadata['benchmarks'][self.benchmark_name]['data']["num_subjects"] == 30


@pytest.mark.integration
class TestBenchmarkMetadataGeneratorEngineeringIntegration:
    """Integration tests for engineering benchmarks using real VisionDomainPlugin."""
    
    def setup_method(self):
        try:
            from brainscore_vision.plugin_management import VisionDomainPlugin
            
            # Try to find the objectnet benchmark in local vision directory structure
            import os
            local_benchmark_path = os.path.join(os.getcwd(), "vision", "brainscore_vision", "benchmarks", "objectnet")
            
            if os.path.exists(local_benchmark_path):
                self.benchmark_path = local_benchmark_path
            else:
                # Fallback to the original approach if the local path doesn't exist
                self.benchmark_path = str(get_installed_package_path("benchmarks/objectnet"))
            
            self.benchmark_name = "ObjectNet-top1"
            self.vision_plugin = VisionDomainPlugin(benchmark_type="engineering")
            self.generator = BenchmarkMetadataGenerator(self.benchmark_path, self.vision_plugin)
            self.benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
            self.yaml_path_created = self.generator(self.benchmark_list)
            self.yaml_path_expected = Path(self.benchmark_path) / "metadata.yml"
        except ImportError:
            pytest.skip("VisionDomainPlugin not available for integration test")

    def teardown_method(self):
        if hasattr(self, 'yaml_path_expected') and self.yaml_path_expected.exists():
            self.yaml_path_expected.unlink()
            print(f"Deleted: {self.yaml_path_expected}")

    def test_benchmark_metadata_methods(self):
        with open(self.yaml_path_expected, 'r') as f:
            metadata = yaml.safe_load(f)
        assert 'benchmarks' in metadata
        assert self.benchmark_name in metadata['benchmarks']

        # Verify real data values for ObjectNet
        assert metadata['benchmarks'][self.benchmark_name]['data']["benchmark_type"] == "engineering"
        assert metadata['benchmarks'][self.benchmark_name]['data']["datatype"] == "engineering"


# =============================================================================
# DOMAIN PLUGIN INTERFACE VALIDATION TESTS  
# =============================================================================

class TestDomainPluginInterface:
    """Test the abstract interface enforcement."""
    
    def test_cannot_instantiate_interface_directly(self):
        """Test that the abstract interface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DomainPluginInterface()

    def test_incomplete_implementation_raises_error(self):
        """Test that incomplete implementations raise TypeError."""
        
        class IncompleteDomainPlugin(DomainPluginInterface):
            def load_benchmark(self, identifier: str):
                pass
            # Missing other required methods
        
        with pytest.raises(TypeError):
            IncompleteDomainPlugin()

    def test_complete_implementation_works(self):
        """Test that complete implementations work properly."""
        mock_plugin = MockDomainPlugin()
        assert isinstance(mock_plugin, DomainPluginInterface)
        
        # Test all required benchmark methods exist and are callable
        assert hasattr(mock_plugin, 'load_benchmark')
        assert hasattr(mock_plugin, 'create_stimuli_metadata')
        assert hasattr(mock_plugin, 'create_data_metadata')
        assert hasattr(mock_plugin, 'create_metric_metadata')
        assert hasattr(mock_plugin, 'find_registered_benchmarks')
        
        # Test all required model methods exist and are callable
        assert hasattr(mock_plugin, 'load_model')
        assert hasattr(mock_plugin, 'find_registered_models')
        assert hasattr(mock_plugin, 'extract_model_for_analysis')
        assert hasattr(mock_plugin, 'detect_model_architecture')
        assert hasattr(mock_plugin, 'get_model_family')
        assert hasattr(mock_plugin, 'create_model_metadata')


if __name__ == "__main__":
    # Run basic unit tests (not integration tests)
    pytest.main([__file__, "-v", "-m", "not integration"])