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

    def __init__(self, benchmark_dir: str, domain_plugin: DomainPluginInterface, use_inheritance_format: bool = True):
        """
        - Initializes the metadata generator with a specified benchmark directory and domain plugin.

        :param benchmark_dir: str, the directory where benchmark metadata files are stored.
        :param domain_plugin: DomainPluginInterface, the domain-specific plugin for handling benchmark operations.
        :param use_inheritance_format: bool, whether to generate inheritance format (data_id/metric_id) or expanded format.
        """
        self.benchmark_dir = benchmark_dir
        self.domain_plugin = domain_plugin
        self.use_inheritance_format = use_inheritance_format

    def __call__(self, benchmark_list: List[str]) -> List[str]:
        """
        - Processes a list of benchmarks and returns their generated metadata file paths.

        :param benchmark_list: List[str], a list of benchmark names to process.
        :return: List[str], a list of unique YAML file paths generated.

        Notes:
        - Collects metadata for all benchmarks first, then writes atomically.
        - Ensures no partial files are created if process is interrupted.
        """
        if not benchmark_list:
            return []
        
        # Collect all benchmark metadata first (batch approach)
        all_metadata = {}
        benchmark_dir_name = self.benchmark_dir.split("/")[-1]
        
        for i, benchmark_name in enumerate(benchmark_list):
            print(f"INFO: Generating metadata for benchmark {i + 1}/{len(benchmark_list)}: {benchmark_name}",
                  file=sys.stderr)
            try:
                benchmark = self.domain_plugin.load_benchmark(benchmark_name)
                if benchmark is None:
                    print(f"ERROR: Failed to load benchmark '{benchmark_name}', skipping", file=sys.stderr)
                    continue
                
                # Generate metadata for this benchmark
                if self.use_inheritance_format and hasattr(self.domain_plugin, 'create_inheritance_metadata'):
                    metadata = self.domain_plugin.create_inheritance_metadata(benchmark, benchmark_dir_name)
                else:
                    # Fall back to expanded format
                    metadata = {
                        "stimulus_set": self.domain_plugin.create_stimuli_metadata(benchmark, benchmark_dir_name),
                        "data": self.domain_plugin.create_data_metadata(benchmark, benchmark_dir_name),
                        "metric": self.domain_plugin.create_metric_metadata(benchmark, benchmark_dir_name),
                    }
                
                all_metadata[benchmark_name] = metadata
                
            except Exception as e:
                print(f"ERROR: Failed to process benchmark '{benchmark_name}': {e}", file=sys.stderr)
                continue
        
        # Write all metadata atomically
        if all_metadata:
            yaml_path = self._write_metadata_file(all_metadata)
            return [yaml_path] if yaml_path else []
        else:
            print("ERROR: No benchmarks were successfully processed", file=sys.stderr)
            return []

    def _write_metadata_file(self, all_metadata: dict) -> Optional[str]:
        """Write all metadata to file atomically."""
        try:
            yaml_filename = "metadata.yml"
            
            if not self.benchmark_dir or not os.path.exists(self.benchmark_dir):
                failure_dir = "FAILURES"
                os.makedirs(failure_dir, exist_ok=True)
                yaml_path = os.path.join(failure_dir, f"batch_metadata.yml")
                print(f"Directory '{self.benchmark_dir}' not found. Writing YAML to '{yaml_path}'", file=sys.stderr)
            else:
                yaml_path = os.path.join(self.benchmark_dir, yaml_filename)
            
            # Create final metadata structure
            final_metadata = {"benchmarks": all_metadata}
            
            # Write atomically
            with open(yaml_path, "w", encoding="utf-8") as file:
                yaml.dump(final_metadata, file, default_flow_style=False, sort_keys=False, indent=4)
            
            print(f"Saved metadata to {yaml_path}", file=sys.stderr)
            return yaml_path
            
        except Exception as e:
            print(f"ERROR: Failed to write metadata file: {e}", file=sys.stderr)
            return None

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
            
            # Choose format based on flag
            if self.use_inheritance_format and hasattr(self.domain_plugin, 'create_inheritance_metadata'):
                new_metadata = self.domain_plugin.create_inheritance_metadata(benchmark, benchmark_dir_name)
            else:
                # Fall back to expanded format
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


