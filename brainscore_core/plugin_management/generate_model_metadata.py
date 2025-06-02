import os
import yaml
from brainscore_core.plugin_management.domain_plugin_interface import DomainPluginInterface
from typing import List, Optional, Any
import sys


class ModelMetadataGenerator:
    """
    - Generates metadata for machine learning models by extracting architecture, family, parameter counts, and more.

    This class provides utilities for discovering, loading, processing, and analyzing ML models using
    domain-specific plugins. It supports metadata extraction, YAML generation, and integration with
    external platforms like Hugging Face and Brain-Score.

    Attributes:
    - plugin_dir (str): The directory where plugin-related metadata is stored.
    - domain_plugin (DomainPluginInterface): The domain-specific plugin for loading models and creating metadata.

    Methods:
    - __call__(model_list: List[str]) -> List[str]: Processes multiple models and returns unique YAML paths.
    - create_yaml(model: Any, model_name: str) -> Optional[str]: Generates a YAML metadata file.
    - process_single_model(model_name: str) -> Optional[str]: Processes a single model.

    Notes:
    - Uses domain-specific plugins to handle model loading and metadata creation.
    - Handles metadata extraction, storage, and linking to external sources.
    - All error handling is printed to prevent silent failures.
    """

    def __init__(self, plugin_dir: str, domain_plugin: DomainPluginInterface):
        """
        - Initializes the metadata generator with a specified plugin directory and domain plugin.

        :param plugin_dir: str, the directory where model metadata files are stored.
        :param domain_plugin: DomainPluginInterface, the domain-specific plugin for handling model operations.
        """
        self.plugin_dir = plugin_dir
        self.domain_plugin = domain_plugin

    def __call__(self, model_list: List[str]) -> List[str]:
        """
        - Processes a list of models and returns their generated metadata file paths.

        :param model_list: List[str], a list of model names to process.
        :return: List[str], a list of unique YAML file paths generated.

        Notes:
        - Calls `process_single_model` for each model in the list.
        - Ensures duplicate YAML paths are removed.
        """
        yaml_paths = set()
        for i, model_name in enumerate(model_list):
            print(f"INFO: Generating metadata for model {i + 1}/{len(model_list)}: {model_name}", file=sys.stderr)
            yaml_path = self.process_single_model(model_name)
            if yaml_path:
                yaml_paths.add(yaml_path)
        return list(yaml_paths)

    def find_registered_models(self, root_folder: str) -> List[str]:
        """
        - Finds all registered models using the domain plugin.

        :param root_folder: str, the root directory to search for model registrations.
        :return: List[str], a list of model names found.

        Notes:
        - Delegates to the domain plugin for domain-specific model discovery.
        """
        return self.domain_plugin.find_registered_models(root_folder)

    def create_yaml(self, model: Any, model_name: str, model_dir: str) -> Optional[str]:
        """
        - Creates or updates a YAML metadata file for the given model, handling errors gracefully.

        :param model: Any, the model object from which metadata is extracted.
        :param model_name: str, the name of the model being processed.
        :param model_dir: str, the directory where the YAML metadata file should be stored.
        :return: Optional[str], the file path of the generated YAML metadata file, or None if an error occurs.

        Notes:
        - The YAML file (`metadata.yml`) is stored in the specified `model_dir`.
        - If an existing YAML file is found, it is updated rather than overwritten.
        - Delegates metadata creation to the domain plugin.
        - Prints an error message and returns None if an exception occurs.
        """
        try:
            yaml_filename = "metadata.yml"
            yaml_path = os.path.join(model_dir, yaml_filename)

            model_dir_name = model_dir.split("/")[-1]
            new_metadata = self.domain_plugin.create_model_metadata(model, model_name, model_dir_name)
            
            if os.path.exists(yaml_path) and os.path.getsize(yaml_path) > 0:
                with open(yaml_path, "r", encoding="utf-8") as file:
                    existing_metadata = yaml.safe_load(file) or {}
            else:
                existing_metadata = {}
            
            if "models" not in existing_metadata:
                existing_metadata["models"] = {}
            
            existing_metadata["models"][model_name] = new_metadata
            
            with open(yaml_path, "w", encoding="utf-8") as file:
                yaml.dump(existing_metadata, file, default_flow_style=False, sort_keys=False, indent=4)
            
            print(f"Saved model metadata to {yaml_path}", file=sys.stderr)
            return yaml_path
        except Exception as e:
            error_message = f"ERROR: Failed to create YAML for '{model_name}': {e}"
            print(error_message, file=sys.stderr)
            return None

    def process_single_model(self, model_name: str) -> Optional[str]:
        """
        - Processes a single model by loading it and generating a YAML metadata file.

        :param model_name: str, the name of the model to be processed.
        :return: Optional[str], the file path of the generated YAML metadata file, or None if the model fails to load.

        Notes:
        - If the model fails to load, the function prints an error and returns None.
        - Uses the domain plugin to load and extract the model for analysis.
        - Any unexpected exceptions are caught and logged.
        """
        try:
            model = self.domain_plugin.load_model(model_name)
            if model is None:
                return None
            
            # Extract the underlying model for analysis using domain-specific logic
            analysis_model = self.domain_plugin.extract_model_for_analysis(model)
            yaml_path = self.create_yaml(analysis_model, model_name, self.plugin_dir)
            return yaml_path
        except Exception as e:
            error_message = f"ERROR: Unexpected error processing '{model_name}': {e}"
            print(error_message, file=sys.stderr)
            return None

