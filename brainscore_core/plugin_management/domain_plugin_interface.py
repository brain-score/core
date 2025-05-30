from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List


class DomainPluginInterface(ABC):
    """
    Abstract interface for domain-specific plugin operations.
    
    This class defines the contract for domain-specific implementations
    that handle loading benchmarks/models and creating metadata for different
    domains (e.g., vision, language, etc.).
    """

    # ============================================================================
    # BENCHMARK-RELATED METHODS
    # ============================================================================

    @abstractmethod
    def load_benchmark(self, identifier: str) -> Optional[object]:
        """
        Load a benchmark using domain-specific logic.
        
        :param identifier: str, the unique name of the benchmark to load.
        :return: Optional[object], the benchmark instance if successfully loaded, otherwise None.
        """
        raise NotImplementedError()

    @abstractmethod
    def create_stimuli_metadata(self, plugin: Any, plugin_dir_name: str) -> Dict[str, Any]:
        """
        Create stimuli metadata for the given plugin.
        
        :param plugin: The benchmark plugin instance.
        :param plugin_dir_name: str, name of the plugin directory.
        :return: Dict[str, Any], stimuli metadata dictionary.
        """
        raise NotImplementedError()

    @abstractmethod
    def create_data_metadata(self, benchmark: Any, benchmark_dir_name: str) -> Dict[str, Any]:
        """
        Create data metadata for the given benchmark.
        
        :param benchmark: The benchmark instance.
        :param benchmark_dir_name: str, name of the benchmark directory.
        :return: Dict[str, Any], data metadata dictionary.
        """
        raise NotImplementedError()

    @abstractmethod
    def create_metric_metadata(self, plugin: Any, plugin_dir_name: str) -> Dict[str, Any]:
        """
        Create metric metadata for the given plugin.
        
        :param plugin: The benchmark plugin instance.
        :param plugin_dir_name: str, name of the plugin directory.
        :return: Dict[str, Any], metric metadata dictionary.
        """
        raise NotImplementedError()

    @abstractmethod
    def find_registered_benchmarks(self, root_folder: str) -> List[str]:
        """
        Find all registered benchmarks in the domain-specific format.
        
        :param root_folder: str, the root directory to search for benchmark registrations.
        :return: List[str], a list of benchmark names found.
        """
        raise NotImplementedError()

    # ============================================================================
    # MODEL-RELATED METHODS
    # ============================================================================

    @abstractmethod
    def load_model(self, identifier: str) -> Optional[object]:
        """
        Load a model using domain-specific logic.
        
        :param identifier: str, the unique name of the model to load.
        :return: Optional[object], the model instance if successfully loaded, otherwise None.
        """
        raise NotImplementedError()

    @abstractmethod
    def find_registered_models(self, root_folder: str) -> List[str]:
        """
        Find all registered models in the domain-specific format.
        
        :param root_folder: str, the root directory to search for model registrations.
        :return: List[str], a list of model names found.
        """
        raise NotImplementedError()

    @abstractmethod
    def extract_model_for_analysis(self, model: Any) -> Any:
        """
        Extract the underlying model from domain-specific wrapper for analysis.
        
        :param model: The domain-specific model wrapper/instance.
        :return: Any, the underlying model suitable for architecture analysis.
        """
        raise NotImplementedError()

    @abstractmethod
    def detect_model_architecture(self, model: Any, model_name: str) -> str:
        """
        Detect the architecture type(s) of the model using domain-specific logic.
        
        :param model: The model instance to analyze.
        :param model_name: str, the name of the model for additional pattern matching.
        :return: str, comma-separated architecture types (e.g., "DCNN", "Transformer").
        """
        raise NotImplementedError()

    @abstractmethod
    def get_model_family(self, model_name: str) -> Optional[str]:
        """
        Extract the model family from the model name using domain-specific patterns.
        
        :param model_name: str, the name of the model to analyze.
        :return: Optional[str], the detected model family/families, or None if unknown.
        """
        raise NotImplementedError()

    @abstractmethod
    def create_model_metadata(self, model: Any, model_name: str, model_dir_name: str) -> Dict[str, Any]:
        """
        Create comprehensive metadata for the given model.
        
        :param model: The model instance.
        :param model_name: str, the name of the model.
        :param model_dir_name: str, name of the model directory.
        :return: Dict[str, Any], model metadata dictionary.
        """
        raise NotImplementedError() 