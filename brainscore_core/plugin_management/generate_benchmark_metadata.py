import os
import yaml
import torch.nn as nn
import re
import requests
from brainscore_core.plugin_management.import_plugin import import_plugin
from brainscore_core.plugin_management.domain_plugin_interface import DomainPluginInterface
from typing import List, Optional, Any
import sys


class BenchmarkMetadataGenerator:
    """
    - Generates metadata for machine learning benchmarks by extracting architecture, family, parameter counts, and more.

    This class provides utilities for discovering, loading, processing, and analyzing ML benchmarks using
    domain-specific plugins. It supports metadata extraction, YAML generation, and integration with
    external platforms like Hugging Face and Brain-Score.

    Attributes:
    - benchmark_dir (str): The directory where benchmark metadata files are stored.
    - domain_plugin (DomainPluginInterface): The domain-specific plugin for loading benchmarks and creating metadata.

    Methods:
    - __call__(benchmark: List[str]) -> List[str]: Processes multiple benchmarks and returns unique YAML paths.
    - create_yaml(benchmark: Any, benchmark_name: str) -> Optional[str]: Generates a YAML metadata file.
    - process_single_benchmark(benchmark_name: str) -> Optional[str]: Processes a single benchmark.

    Notes:
    - Uses domain-specific plugins to handle benchmark loading and metadata creation.
    - Handles metadata extraction, storage, and linking to external sources.
    - All error handling is printed to prevent silent failures.
    """

    def __init__(self, benchmark_dir: str, domain_plugin: DomainPluginInterface):
        """
        - Initializes the metadata generator with a specified benchmark directory and domain plugin.

        :param benchmark_dir: str, the directory where benchmark metadata files are stored.
        :param domain_plugin: DomainPluginInterface, the domain-specific plugin for handling benchmark operations.
        """
        self.benchmark_dir = benchmark_dir
        self.domain_plugin = domain_plugin

    def __call__(self, benchmark_list: List[str]) -> List[str]:
        """
        - Processes a list of benchmarks and returns their generated metadata file paths.

        :param benchmark_list: List[str], a list of benchmark names to process.
        :return: List[str], a list of unique YAML file paths generated.

        Notes:
        - Calls `process_single_benchmark` for each benchmark in the list.
        - Ensures duplicate YAML paths are removed.
        """
        yaml_paths = set()
        for i, benchmark_name in enumerate(benchmark_list):
            print(f"INFO: Generating metadata for benchmark {i + 1}/{len(benchmark_list)}: {benchmark_name}",
                  file=sys.stderr)
            yaml_path = self.process_single_benchmark(benchmark_name)
            if yaml_path:
                yaml_paths.add(yaml_path)
        return list(yaml_paths)

    def create_yaml(self, benchmark, benchmark_name: str, benchmark_dir: str):
        """Create or update YAML metadata for the benchmark, handling errors gracefully."""
        try:
            yaml_filename = "metadata.yml"

            if not benchmark_dir or not os.path.exists(benchmark_dir):
                failure_dir = "FAILURES"
                os.makedirs(failure_dir, exist_ok=True)
                yaml_path = os.path.join(failure_dir, f"{benchmark_name}_metadata.yml")
                print("Directory '{plugin_dir}' not found. Writing YAML to '{yaml_path}'", file=sys.stderr)
            else:
                yaml_path = os.path.join(benchmark_dir, yaml_filename)

            benchmark_dir_name = benchmark_dir.split("/")[-1]
            new_metadata = {
                "stimulus_set": self.domain_plugin.create_stimuli_metadata(benchmark, benchmark_dir_name),
                "data": self.domain_plugin.create_data_metadata(benchmark, benchmark_dir_name),
                "metric": self.domain_plugin.create_metric_metadata(benchmark, benchmark_dir_name),
            }
            if os.path.exists(yaml_path) and os.path.getsize(yaml_path) > 0:
                with open(yaml_path, "r", encoding="utf-8") as file:
                    existing_metadata = yaml.safe_load(file) or {}
            else:
                existing_metadata = {}
            if "benchmarks" not in existing_metadata:
                existing_metadata["benchmarks"] = {}
            existing_metadata["benchmarks"][benchmark_name] = new_metadata
            with open(yaml_path, "w", encoding="utf-8") as file:
                yaml.dump(existing_metadata, file, default_flow_style=False, sort_keys=False, indent=4)
            print(f"Saved metadata to {yaml_path}", file=sys.stderr)
            return yaml_path
        except Exception as e:
            error_message = f"Failed to create YAML for '{benchmark_name}': {e}"
            print(error_message, file=sys.stderr)

    def process_single_benchmark(self, benchmark_name: str) -> Optional[str]:
        """
        - Processes a single benchmark by loading it and generating a YAML metadata file.

        :param benchmark_name: str, the name of the benchmark to be processed.
        :return: Optional[str], the file path of the generated YAML metadata file, or None if the benchmark fails to load.

        Notes:
        - If the benchmark fails to load, the function prints an error and returns None.
        - Any unexpected exceptions are caught and logged.
        """
        try:
            benchmark = self.domain_plugin.load_benchmark(benchmark_name)
            if benchmark is None:
                return
            yaml_path = self.create_yaml(benchmark, benchmark_name, self.benchmark_dir)
            return yaml_path
        except Exception as e:
            error_message = f"ERROR: Unexpected error processing '{benchmark_name}': {e}"
            print(error_message, file=sys.stderr)
            return None

    def find_registered_benchmarks(self, root_folder: str) -> List[str]:
        """
        - Finds all registered benchmarks using the domain plugin.

        :param root_folder: str, the root directory to search for benchmark registrations.
        :return: List[str], a list of benchmark names found.

        Notes:
        - Delegates to the domain plugin for domain-specific benchmark discovery.
        """
        return self.domain_plugin.find_registered_benchmarks(root_folder)


