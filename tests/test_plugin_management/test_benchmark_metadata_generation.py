import os
import tempfile
import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock
from brainscore_core.plugin_management.generate_benchmark_metadata import BenchmarkMetadataGenerator
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


class MockBenchmarkDomainPlugin(DomainPluginInterface):
    """Mock domain plugin for benchmark testing with support for all benchmark types."""
    
    def __init__(self, benchmark_type="neural"):
        self.benchmark_type = benchmark_type
        self.mock_benchmark = Mock()
        self.mock_benchmark._assembly = Mock()
        self.mock_benchmark._assembly.stimulus_set = Mock()
        
        # Setup mock data based on benchmark type
        self.setup_mock_data()
    
    def setup_mock_data(self):
        """Setup mock data based on benchmark type."""
        stimulus_set = self.mock_benchmark._assembly.stimulus_set
        
        if self.benchmark_type == "neural":
            stimulus_set.__len__ = Mock(return_value=616)
            stimulus_set.memory_usage = Mock(return_value=Mock())
            stimulus_set.memory_usage.return_value.sum = Mock(return_value=3758038)  # ~3.6MB
            
            # Neural assembly data
            assembly = self.mock_benchmark._assembly
            assembly.hemisphere = Mock()
            assembly.hemisphere.values = ["L", "L", "L"]
            assembly.subject = Mock() 
            assembly.subject.values = ["sub1", "sub2", "sub1", "sub2"]
            assembly.region = Mock()
            assembly.region.values = ["IT", "IT", "IT"]
            
        elif self.benchmark_type == "behavioral":
            stimulus_set.__len__ = Mock(return_value=22560)
            stimulus_set.memory_usage = Mock(return_value=Mock())
            stimulus_set.memory_usage.return_value.sum = Mock(return_value=13274603)  # ~12.66MB
            
            # Behavioral assembly data (no brain regions)
            assembly = self.mock_benchmark._assembly
            assembly.hemisphere = Mock()
            assembly.hemisphere.values = []
            assembly.subject = Mock() 
            assembly.subject.values = [f"sub{i}" for i in range(30)]  # 30 subjects
            assembly.region = Mock()
            assembly.region.values = []
            
        else:  # engineering
            stimulus_set.__len__ = Mock(return_value=None)
            stimulus_set.memory_usage = Mock(return_value=Mock())
            stimulus_set.memory_usage.return_value.sum = Mock(return_value=None)
            
            # Engineering assembly data (minimal)
            assembly = self.mock_benchmark._assembly
            assembly.hemisphere = Mock()
            assembly.hemisphere.values = []
            assembly.subject = Mock() 
            assembly.subject.values = []
            assembly.region = Mock()
            assembly.region.values = []
    
    def load_benchmark(self, identifier: str):
        """Mock benchmark loading for all types."""
        valid_benchmarks = [
            "test_benchmark", "Rajalingham2020.IT-pls", 
            "tong.Coggan2024_behavior-ConditionWiseAccuracySimilarity", "ObjectNet-top1"
        ]
        if any(bench in identifier for bench in valid_benchmarks):
            return self.mock_benchmark
        return None
    
    def create_stimuli_metadata(self, plugin, plugin_dir_name: str):
        """Mock stimuli metadata creation for all benchmark types."""
        if self.benchmark_type == "neural":
            return {
                "num_stimuli": 616,
                "datatype": "image",
                "stimuli_subtype": None,
                "total_size_mb": 3.5858,
                "brainscore_link": f"https://github.com/brain-score/vision/tree/master/brainscore_vision/data/{plugin_dir_name}",
                "extra_notes": None
            }
        elif self.benchmark_type == "behavioral":
            return {
                "num_stimuli": 22560,
                "datatype": "image", 
                "stimuli_subtype": None,
                "total_size_mb": 12.6584,
                "brainscore_link": f"https://github.com/brain-score/vision/tree/master/brainscore_vision/data/{plugin_dir_name}",
                "extra_notes": None
            }
        else:  # engineering
            return {
                "num_stimuli": None,
                "datatype": "image",
                "stimuli_subtype": None,
                "total_size_mb": None,
                "brainscore_link": f"https://github.com/brain-score/vision/tree/master/brainscore_vision/data/{plugin_dir_name}",
                "extra_notes": None
            }
    
    def create_data_metadata(self, benchmark, benchmark_dir_name: str):
        """Mock data metadata creation for all benchmark types."""
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
        """Mock metric metadata creation for all benchmark types."""
        return {
            "type": None,
            "reference": None,
            "public": None,
            "brainscore_link": f"https://github.com/brain-score/vision/tree/master/brainscore_vision/benchmarks/{plugin_dir_name}",
            "extra_notes": None
        }
    
    def find_registered_benchmarks(self, root_folder: str):
        """Mock benchmark discovery for all types."""
        if "rajalingham2020" in root_folder:
            return ["Rajalingham2020.IT-pls"]
        elif "coggan2024_behavior" in root_folder:
            return ["tong.Coggan2024_behavior-ConditionWiseAccuracySimilarity"]
        elif "objectnet" in root_folder:
            return ["ObjectNet-top1"]
        else:
            return ["test_benchmark"]

    # Model methods (minimal implementation for interface compliance)
    def load_model(self, identifier: str): return None
    def find_registered_models(self, root_folder: str): return []
    def extract_model_for_analysis(self, model): return None
    def detect_model_architecture(self, model, model_name: str): return "DCNN"
    def get_model_family(self, model_name: str): return None
    def create_model_metadata(self, model, model_name: str, model_dir_name: str): return {}


# =============================================================================
# BENCHMARK METADATA GENERATION - FILE OPERATIONS TESTS
# =============================================================================

class TestBenchmarkMetadataFileOperations:
    """Tests focused on benchmark metadata file creation and structure across all benchmark types."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.benchmark_path = self.temp_dir
        self.benchmark_name = "test_benchmark"

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_neural_yaml_file_creation(self):
        """Test YAML file creation for neural benchmarks."""
        mock_plugin = MockBenchmarkDomainPlugin(benchmark_type="neural")
        generator = BenchmarkMetadataGenerator(self.benchmark_path, mock_plugin)
        
        benchmark_list = generator.find_registered_benchmarks(self.benchmark_path)
        yaml_paths = generator(benchmark_list)
        
        yaml_path = yaml_paths[0] if yaml_paths else None
        expected_path = Path(self.benchmark_path) / "metadata.yml"
        
        assert Path(yaml_path).exists()
        assert Path(yaml_path) == expected_path

    def test_behavioral_yaml_file_creation(self):
        """Test YAML file creation for behavioral benchmarks."""
        mock_plugin = MockBenchmarkDomainPlugin(benchmark_type="behavioral")
        generator = BenchmarkMetadataGenerator(self.benchmark_path, mock_plugin)
        
        benchmark_list = generator.find_registered_benchmarks(self.benchmark_path)
        yaml_paths = generator(benchmark_list)
        
        yaml_path = yaml_paths[0] if yaml_paths else None
        expected_path = Path(self.benchmark_path) / "metadata.yml"
        
        assert Path(yaml_path).exists()
        assert Path(yaml_path) == expected_path

    def test_engineering_yaml_file_creation(self):
        """Test YAML file creation for engineering benchmarks."""
        mock_plugin = MockBenchmarkDomainPlugin(benchmark_type="engineering")
        generator = BenchmarkMetadataGenerator(self.benchmark_path, mock_plugin)
        
        benchmark_list = generator.find_registered_benchmarks(self.benchmark_path)
        yaml_paths = generator(benchmark_list)
        
        yaml_path = yaml_paths[0] if yaml_paths else None
        expected_path = Path(self.benchmark_path) / "metadata.yml"
        
        assert Path(yaml_path).exists()
        assert Path(yaml_path) == expected_path

    def test_yaml_structure_validity_all_types(self):
        """Test that generated YAML has correct structure for all benchmark types."""
        for benchmark_type in ["neural", "behavioral", "engineering"]:
            with self.setup_for_type(benchmark_type) as (generator, temp_path):
                benchmark_list = generator.find_registered_benchmarks(temp_path)
                generator(benchmark_list)
                
                yaml_path = Path(temp_path) / "metadata.yml"
                with open(yaml_path, 'r') as f:
                    metadata = yaml.safe_load(f)
                
                assert 'benchmarks' in metadata
                assert self.benchmark_name in metadata['benchmarks']
                
                benchmark_meta = metadata['benchmarks'][self.benchmark_name]
                assert 'stimulus_set' in benchmark_meta
                assert 'data' in benchmark_meta
                assert 'metric' in benchmark_meta

    def test_required_fields_present_all_types(self):
        """Test that all required benchmark fields are present for all types."""
        for benchmark_type in ["neural", "behavioral", "engineering"]:
            with self.setup_for_type(benchmark_type) as (generator, temp_path):
                benchmark_list = generator.find_registered_benchmarks(temp_path)
                generator(benchmark_list)
                
                yaml_path = Path(temp_path) / "metadata.yml"
                with open(yaml_path, 'r') as f:
                    metadata = yaml.safe_load(f)
                
                benchmark_meta = metadata['benchmarks'][self.benchmark_name]
                
                # Required stimulus_set fields for all types
                stimulus_fields = ["num_stimuli", "datatype", "stimuli_subtype", "total_size_mb", "brainscore_link", "extra_notes"]
                for field in stimulus_fields:
                    assert field in benchmark_meta['stimulus_set'], f"Missing {field} in {benchmark_type} stimulus_set"
                
                # Required data fields for all types
                data_fields = ["benchmark_type", "brainscore_link"]
                for field in data_fields:
                    assert field in benchmark_meta['data'], f"Missing {field} in {benchmark_type} data"

    def setup_for_type(self, benchmark_type):
        """Helper method to setup generator for specific benchmark type."""
        import tempfile
        import contextlib
        
        @contextlib.contextmanager
        def temp_generator():
            temp_dir = tempfile.mkdtemp()
            try:
                mock_plugin = MockBenchmarkDomainPlugin(benchmark_type=benchmark_type)
                generator = BenchmarkMetadataGenerator(temp_dir, mock_plugin)
                yield generator, temp_dir
            finally:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        return temp_generator()


# =============================================================================
# BENCHMARK METADATA GENERATION - DATA EXTRACTION TESTS  
# =============================================================================

class TestNeuralBenchmarkDataExtraction:
    """Tests focused on extracting neural-specific metadata values."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.benchmark_path = self.temp_dir
        self.benchmark_name = "test_benchmark"
        self.mock_plugin = MockBenchmarkDomainPlugin(benchmark_type="neural")
        self.generator = BenchmarkMetadataGenerator(self.benchmark_path, self.mock_plugin)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_neural_benchmark_type_extraction(self):
        """Test that benchmark type is correctly identified as neural."""
        benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        self.generator(benchmark_list)
        
        yaml_path = Path(self.benchmark_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        assert metadata['benchmarks'][self.benchmark_name]['data']["benchmark_type"] == "neural"

    def test_brain_region_extraction(self):
        """Test that brain region is correctly extracted from neural benchmarks."""
        benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        self.generator(benchmark_list)
        
        yaml_path = Path(self.benchmark_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        assert metadata['benchmarks'][self.benchmark_name]['data']["region"] == "IT"

    def test_hemisphere_extraction(self):
        """Test that hemisphere is correctly extracted from neural benchmarks."""
        benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        self.generator(benchmark_list)
        
        yaml_path = Path(self.benchmark_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        assert metadata['benchmarks'][self.benchmark_name]['data']["hemisphere"] == "L"

    def test_neural_subject_count_extraction(self):
        """Test that subject count is correctly extracted from neural benchmarks."""
        benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        self.generator(benchmark_list)
        
        yaml_path = Path(self.benchmark_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        assert metadata['benchmarks'][self.benchmark_name]['data']["num_subjects"] == 2

    def test_neural_stimuli_count_extraction(self):
        """Test that stimulus count is correctly extracted from neural benchmarks."""
        benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        self.generator(benchmark_list)
        
        yaml_path = Path(self.benchmark_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["num_stimuli"] == 616

    def test_neural_datatype_null(self):
        """Test that neural benchmarks have null datatype (not behavioral/engineering)."""
        benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        self.generator(benchmark_list)
        
        yaml_path = Path(self.benchmark_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        assert metadata['benchmarks'][self.benchmark_name]['data']["datatype"] is None


class TestBehavioralBenchmarkDataExtraction:
    """Tests focused on extracting behavioral-specific metadata values."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.benchmark_path = self.temp_dir
        self.benchmark_name = "test_benchmark"
        self.mock_plugin = MockBenchmarkDomainPlugin(benchmark_type="behavioral")
        self.generator = BenchmarkMetadataGenerator(self.benchmark_path, self.mock_plugin)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_behavioral_benchmark_type_extraction(self):
        """Test that benchmark type is correctly identified as behavioral."""
        benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        self.generator(benchmark_list)
        
        yaml_path = Path(self.benchmark_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        assert metadata['benchmarks'][self.benchmark_name]['data']["benchmark_type"] == "behavioral"

    def test_behavioral_datatype_classification(self):
        """Test that behavioral benchmarks are classified with behavioral datatype."""
        benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        self.generator(benchmark_list)
        
        yaml_path = Path(self.benchmark_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        assert metadata['benchmarks'][self.benchmark_name]['data']["datatype"] == "behavioral"

    def test_behavioral_subject_count_extraction(self):
        """Test that subject count is correctly extracted from behavioral benchmarks."""
        benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        self.generator(benchmark_list)
        
        yaml_path = Path(self.benchmark_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        assert metadata['benchmarks'][self.benchmark_name]['data']["num_subjects"] == 30

    def test_behavioral_stimuli_count_extraction(self):
        """Test that stimulus count is correctly extracted from behavioral benchmarks."""
        benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        self.generator(benchmark_list)
        
        yaml_path = Path(self.benchmark_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["num_stimuli"] == 22560

    def test_behavioral_null_brain_region_handling(self):
        """Test that behavioral benchmarks have null brain region."""
        benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        self.generator(benchmark_list)
        
        yaml_path = Path(self.benchmark_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        assert metadata['benchmarks'][self.benchmark_name]['data']["region"] is None

    def test_behavioral_null_hemisphere_handling(self):
        """Test that behavioral benchmarks have null hemisphere."""
        benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        self.generator(benchmark_list)
        
        yaml_path = Path(self.benchmark_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        assert metadata['benchmarks'][self.benchmark_name]['data']["hemisphere"] is None


class TestEngineeringBenchmarkDataExtraction:
    """Tests focused on extracting engineering-specific metadata values."""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.benchmark_path = self.temp_dir
        self.benchmark_name = "test_benchmark"
        self.mock_plugin = MockBenchmarkDomainPlugin(benchmark_type="engineering")
        self.generator = BenchmarkMetadataGenerator(self.benchmark_path, self.mock_plugin)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_engineering_benchmark_type_extraction(self):
        """Test that benchmark type is correctly identified as engineering."""
        benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        self.generator(benchmark_list)
        
        yaml_path = Path(self.benchmark_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        assert metadata['benchmarks'][self.benchmark_name]['data']["benchmark_type"] == "engineering"

    def test_engineering_datatype_classification(self):
        """Test that engineering benchmarks are classified with engineering datatype."""
        benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        self.generator(benchmark_list)
        
        yaml_path = Path(self.benchmark_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        assert metadata['benchmarks'][self.benchmark_name]['data']["datatype"] == "engineering"

    def test_engineering_null_subject_handling(self):
        """Test that engineering benchmarks have null subject count."""
        benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        self.generator(benchmark_list)
        
        yaml_path = Path(self.benchmark_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        assert metadata['benchmarks'][self.benchmark_name]['data']["num_subjects"] is None

    def test_engineering_null_stimuli_handling(self):
        """Test that engineering benchmarks handle null stimulus counts."""
        benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        self.generator(benchmark_list)
        
        yaml_path = Path(self.benchmark_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        assert metadata['benchmarks'][self.benchmark_name]['stimulus_set']["num_stimuli"] is None


# =============================================================================
# BENCHMARK METADATA GENERATION - INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
class TestNeuralBenchmarkIntegration:
    """Integration tests with real neural benchmarks."""
    
    def setup_method(self):
        try:
            from brainscore_vision.plugin_management import VisionDomainPlugin
            
            import os
            local_benchmark_path = os.path.join(os.getcwd(), "vision", "brainscore_vision", "benchmarks", "rajalingham2020")
            
            if os.path.exists(local_benchmark_path):
                self.benchmark_path = local_benchmark_path
            else:
                self.benchmark_path = str(get_installed_package_path("benchmarks/rajalingham2020"))
            
            self.benchmark_name = "Rajalingham2020.IT-pls"
            self.vision_plugin = VisionDomainPlugin(benchmark_type="neural")
            self.generator = BenchmarkMetadataGenerator(self.benchmark_path, self.vision_plugin)
        except ImportError:
            pytest.skip("VisionDomainPlugin not available for integration test")

    def teardown_method(self):
        yaml_path = Path(self.benchmark_path) / "metadata.yml"
        if yaml_path.exists():
            yaml_path.unlink()

    def test_real_rajalingham2020_metadata_extraction(self):
        """Test metadata extraction from real Rajalingham2020 neural benchmark."""
        benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        yaml_paths = self.generator(benchmark_list)
        
        yaml_path = Path(self.benchmark_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        benchmark_meta = metadata['benchmarks'][self.benchmark_name]
        
        # Verify real Rajalingham2020 values
        assert benchmark_meta['stimulus_set']["num_stimuli"] == 616
        assert benchmark_meta['stimulus_set']["datatype"] == "image"
        assert benchmark_meta['data']["benchmark_type"] == "neural"
        assert benchmark_meta['data']["region"] == "IT"
        assert benchmark_meta['data']["hemisphere"] == "L"
        assert benchmark_meta['data']["datatype"] is None


@pytest.mark.integration 
class TestBehavioralBenchmarkIntegration:
    """Integration tests with real behavioral benchmarks."""
    
    def setup_method(self):
        try:
            from brainscore_vision.plugin_management import VisionDomainPlugin
            
            import os
            local_benchmark_path = os.path.join(os.getcwd(), "vision", "brainscore_vision", "benchmarks", "coggan2024_behavior")
            
            if os.path.exists(local_benchmark_path):
                self.benchmark_path = local_benchmark_path
            else:
                self.benchmark_path = str(get_installed_package_path("benchmarks/coggan2024_behavior"))
            
            self.benchmark_name = "tong.Coggan2024_behavior-ConditionWiseAccuracySimilarity"
            self.vision_plugin = VisionDomainPlugin(benchmark_type="behavioral")
            self.generator = BenchmarkMetadataGenerator(self.benchmark_path, self.vision_plugin)
        except ImportError:
            pytest.skip("VisionDomainPlugin not available for integration test")

    def teardown_method(self):
        yaml_path = Path(self.benchmark_path) / "metadata.yml"
        if yaml_path.exists():
            yaml_path.unlink()

    def test_real_coggan2024_metadata_extraction(self):
        """Test metadata extraction from real Coggan2024 behavioral benchmark."""
        benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        yaml_paths = self.generator(benchmark_list)
        
        yaml_path = Path(self.benchmark_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        benchmark_meta = metadata['benchmarks'][self.benchmark_name]
        
        # Verify real Coggan2024 values
        assert benchmark_meta['data']["benchmark_type"] == "behavioral"
        assert benchmark_meta['data']["datatype"] == "behavioral"
        assert benchmark_meta['data']["num_subjects"] == 30
        assert benchmark_meta['data']["region"] is None
        assert benchmark_meta['data']["hemisphere"] is None


@pytest.mark.integration
class TestEngineeringBenchmarkIntegration:
    """Integration tests with real engineering benchmarks."""
    
    def setup_method(self):
        try:
            from brainscore_vision.plugin_management import VisionDomainPlugin
            
            import os
            local_benchmark_path = os.path.join(os.getcwd(), "vision", "brainscore_vision", "benchmarks", "objectnet")
            
            if os.path.exists(local_benchmark_path):
                self.benchmark_path = local_benchmark_path
            else:
                self.benchmark_path = str(get_installed_package_path("benchmarks/objectnet"))
            
            self.benchmark_name = "ObjectNet-top1"
            self.vision_plugin = VisionDomainPlugin(benchmark_type="engineering")
            self.generator = BenchmarkMetadataGenerator(self.benchmark_path, self.vision_plugin)
        except ImportError:
            pytest.skip("VisionDomainPlugin not available for integration test")

    def teardown_method(self):
        yaml_path = Path(self.benchmark_path) / "metadata.yml"
        if yaml_path.exists():
            yaml_path.unlink()

    def test_real_objectnet_metadata_extraction(self):
        """Test metadata extraction from real ObjectNet engineering benchmark."""
        benchmark_list = self.generator.find_registered_benchmarks(self.benchmark_path)
        yaml_paths = self.generator(benchmark_list)
        
        yaml_path = Path(self.benchmark_path) / "metadata.yml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        benchmark_meta = metadata['benchmarks'][self.benchmark_name]
        
        # Verify real ObjectNet values
        assert benchmark_meta['data']["benchmark_type"] == "engineering"
        assert benchmark_meta['data']["datatype"] == "engineering"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"]) 