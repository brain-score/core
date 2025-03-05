import os
import yaml
import torch.nn as nn
import re
import requests
from brainscore_core.plugin_management.import_plugin import import_plugin
from typing import List, Optional, Any
import sys


class ModelMetadataGenerator:
    """
    - Generates metadata for machine learning models by extracting architecture, family, parameter counts, and more.

    This class provides utilities for discovering, loading, processing, and analyzing ML models, primarily using
    the `brainscore_vision` framework. It supports metadata extraction, YAML generation, and integration with
    external platforms like Hugging Face and Brain-Score.

    Attributes:
    - plugin_dir (str): The directory where plugin-related metadata is stored.

    Methods:
    - __call__(model_list: List[str]) -> List[str]: Processes multiple models and returns unique YAML paths.
    - find_registered_models(root_folder: str) -> List[str]: Finds registered models in `__init__.py` files.
    - load_model(identifier: str) -> Optional[object]: Loads a model using `brainscore_vision`.
    - detect_model_architecture(model: nn.Module, model_name: str) -> str: Identifies the model's architecture.
    - get_huggingface_link(model_name: str) -> Optional[str]: Checks if a Hugging Face repository exists for the model.
    - get_model_family(model_name: str) -> Optional[str]: Extracts the model family from the name.
    - create_yaml(model: Any, model_name: str) -> Optional[str]: Generates a YAML metadata file.
    - process_single_model(model_name: str) -> Optional[str]: Processes a single model.

    Notes:
    - Designed for models following the `brainscore_vision` plugin structure.
    - Uses regex to classify models based on naming conventions.
    - Handles metadata extraction, storage, and linking to external sources.
    - All error handling is printed to prevent silent failures.
    """

    def __init__(self, plugin_dir: str):
        """
        - Initializes the metadata generator with a specified plugin directory.

        :param plugin_dir: str, the directory where model metadata files are stored.
        """
        self.plugin_dir = plugin_dir

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
        - Finds all registered models inside `__init__.py` files within a given root directory.

        :param root_folder: str, the root directory to search for model registrations.
        :return: List[str], a list of model names found in `model_registry` assignments.

        Notes:
        - Recursively searches for `__init__.py` files in the specified directory.
        - Extracts model names assigned to `model_registry[...] = lambda:` using regex.
        - Logs an error message if any `__init__.py` file cannot be read.
        """

        registered_models = []

        for dirpath, _, filenames in os.walk(root_folder):
            for filename in filenames:
                if filename == "__init__.py":
                    init_file_path = os.path.join(dirpath, filename)
                    try:
                        with open(init_file_path, "r", encoding="utf-8") as file:
                            content = file.read()
                        matches = re.findall(r'model_registry\[\s*["\'](.*?)["\']\s*\]\s*=\s*lambda:', content)
                        if matches:
                            registered_models.extend(matches)
                    except Exception as e:
                        error_message = f"ERROR: Could not read __init__.py file path {init_file_path}: {e}"
                        print(error_message, file=sys.stderr)
        return registered_models

    def load_model(self, identifier: str) -> Optional[object]:
        """
        - Loads a model using `brainscore_vision` and returns the model instance.

        :param identifier: str, the unique name of the model to load.
        :return: Optional[object], the model instance if successfully loaded, otherwise None.

        Notes:
        - Uses `import_plugin` to dynamically load the model from `brainscore_vision.models`.
        - Retrieves the model instance from `model_registry` using the given identifier.
        - Returns `None` if an error occurs during model loading.
        - Prints an error message if the model fails to load.
        """

        try:
            import_plugin('brainscore_vision', 'models', identifier)
            from brainscore_vision import model_registry
            model_instance = model_registry[identifier]()
            return model_instance
        except Exception as e:
            error_message = f"ERROR: Failed to load model '{identifier}': {e}"
            print(error_message, file=sys.stderr)
            return None, None

    def detect_model_architecture(self, model: nn.Module, model_name: str) -> str:
        """
        - Determines whether the model architecture is a DCNN, RNN, Transformer, or a combination.

        :param model: Pytorch nn.Module, the model instance to analyze.
        :param model_name: str, the name of the model, used for additional pattern-based classification.
        :return: str, a comma-separated string representing the detected architecture types.

        Notes:
        - Default classification is "DCNN" (Deep Convolutional Neural Network).
        - Hardcodes `"RNN"` and `"SKIP_CONNECTIONS"` for cornet models based on name matching.
        - Detects Transformers by checking for `MultiheadAttention`, `LayerNorm`, or class names containing
          "transformer" or "attention".
        - Detects RNN-based architectures by checking for instances of `RNN`, `LSTM`, or `GRU`.
        - The returned architecture tags are sorted for consistency.
        """

        tags = {"DCNN"}
        if re.search(r'cor[_-]*net', model_name, re.IGNORECASE):
            tags.add("RNN")
            tags.add("SKIP_CONNECTIONS")  # hardcode cornets
        if any(
                isinstance(layer, (nn.MultiheadAttention, nn.LayerNorm)) or
                'transformer' in layer.__class__.__name__.lower() or
                'attention' in layer.__class__.__name__.lower()
                for layer in model.modules()
        ):
            tags.add("Transformer")
        if any(isinstance(layer, (nn.RNN, nn.LSTM, nn.GRU)) for layer in model.modules()):
            tags.add("RNN")
        return ", ".join(sorted(tags))

    def get_huggingface_link(self, model_name: str) -> Optional[str]:
        """
        - Checks if a Hugging Face model repository exists based on the model name.

        :param model_name: str, the name of the model to check on Hugging Face.
        :return: Optional[str], the Hugging Face repository URL if it exists, otherwise None.

        Notes:
        - The model name is sanitized by replacing `:` and `/` with `-` to match Hugging Face repository naming conventions.
        - Sends a HEAD request to check if the model repository exists.
        - Uses a 1-second timeout to avoid long delays in case of network issues.
        - Prints a warning and returns None if the request fails or the repository does not exist.
        """

        sanitized_model_name = model_name.replace(":", "-").replace("/", "-")
        hf_url = f"https://huggingface.co/{sanitized_model_name}"
        try:
            response = requests.head(hf_url, timeout=1)
            if response.status_code == 200:
                return hf_url
            print(f"HuggingFace link for {model_name} found.", file=sys.stderr)
        except requests.RequestException as e:
            print(f"WARNING: checking HuggingFace link for '{model_name}': {e} failed.", file=sys.stderr)
        return None

    def get_model_family(self, model_name: str) -> Optional[str]:
        """
        - Extracts the model family based on standard well-known architectures.

        :param model_name: str, the name of the model to analyze.
        :return: Optional[str], a comma-separated string of detected model families, or None if no known family is found.

        Notes:
        - Supports multiple families (e.g., 'resnet', 'efficientnet') if the model name contains multiple patterns.
        - Uses regex matching to identify model families, ignoring case sensitivity.
        - Some families are common, others are Brain-Score specific (cornet, vone, etc)
        """
        families = []

        known_families = {
            "resnet": r'resnet',
            "resnext": r'resnext',
            "alexnet": r'alexnet',
            "efficientnet": r'efficientnet|effnet',
            "convnext": r'convnext',
            "vit": r'vit|visiontransformer',
            "densenet": r'densenet',
            "nasnet": r'nasnet',
            "pnasnet": r'pnasnet',
            "inception": r'inception',
            "swin": r'swin',
            "mobilenet": r'mobilenet|mobilevit',
            "mvit": r'mvit',
            "slowfast": r'slowfast',
            "i3d": r'i3d',
            "x3d": r'x3d',
            "timesformer": r'timesformer',
            "s3d": r's3d',
            "r3d": r'r3d',
            "r2plus1d": r'r2plus1d',
            "deit": r'deit',
            "cornet": r'cornet',
            "vgg": r'vgg',
            "clip": r'clip',
            "cvt": r'cvt',
            "vone": r'vone'
        }
        for family, pattern in known_families.items():
            if re.search(pattern, model_name, re.IGNORECASE):
                families.append(family)
        return ", ".join(sorted(families)) if families else None

    def create_yaml(self, model: Any, model_name: str, model_dir: str) -> Optional[str]:
        """
        - Creates or updates a YAML metadata file for the given model, handling errors gracefully.

        :param model: Any, the model object from which metadata is extracted.
        :param model_name: str, the name of the model being processed.
        :param model_dir: str, the directory where the YAML metadata file should be stored.
        :return: Optional[str], the file path of the generated YAML metadata file, or None if an error occurs.

        Notes:
        - The YAML file (`metadata.yaml`) is stored in the specified `model_dir`.
        - If an existing YAML file is found, it is updated rather than overwritten.
        - Extracts model architecture, family, parameter counts, layer details, and model size.
        - Adds links to Brain-Score and Hugging Face, if available.
        - Prints an error message and returns None if an exception occurs.
        """
        try:
            yaml_filename = "metadata.yaml"
            yaml_path = os.path.join(model_dir, yaml_filename)

            model_dir_name = model_dir.split("/")[-1]
            architecture_type = self.detect_model_architecture(model, model_name)

            new_metadata = {
                "architecture": architecture_type,
                "model_family": self.get_model_family(model_name),
                "total_parameter_count": sum(p.numel() for p in model.parameters()),
                "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "total_layers": sum(1 for _ in model.modules()),
                "trainable_layers": sum(1 for p in model.parameters() if p.requires_grad and p.dim() > 1),
                "model_size_MB": round(sum(p.element_size() * p.numel() for p in model.parameters()) / 1e6, 2),
                "training_dataset": None,
                "task_specialization": None,
                "brainscore_link": f"https://github.com/brain-score/vision/tree/master/brainscore_vision/models/{model_dir_name}",
                "huggingface_link": self.get_huggingface_link(model_name),
                "extra_notes": None
            }
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
            return yaml_path
        except Exception as e:
            error_message = f"ERROR: Failed to create YAML for '{model_name}': {e}"
            print(error_message, file=sys.stderr)

    def process_single_model(self, model_name: str) -> Optional[str]:
        """
        - Processes a single model by loading it and generating a YAML metadata file.

        :param model_name: str, the name of the model to be processed.
        :return: Optional[str], the file path of the generated YAML metadata file, or None if the model fails to load.

        Notes:
        - If the model fails to load, the function prints an error and returns None.
        - The function attempts to extract the underlying model from `model.activations_model._model` before processing.
        - Any unexpected exceptions are caught and logged.
        """
        try:
            model = self.load_model(model_name)
            if model is None:
                return
            yaml_path = self.create_yaml(model.activations_model._model, model_name, self.plugin_dir)
            return yaml_path
        except Exception as e:
            error_message = f"ERROR: Unexpected error processing '{model_name}': {e}"
            print(error_message, file=sys.stderr)
            return None

