import os
import yaml
import torch.nn as nn
import re
import requests
from brainscore_core.plugin_management.import_plugin import import_plugin
from typing import List, Optional, Any
import sys


class BenchmarkMetadataGenerator:
    """
    - Generates metadata for machine learning benchmarks by extracting architecture, family, parameter counts, and more.

    This class provides utilities for discovering, loading, processing, and analyzing ML benchmarks, primarily using
    the `brainscore_vision` framework. It supports metadata extraction, YAML generation, and integration with
    external platforms like Hugging Face and Brain-Score.

    Attributes:
    - plugin_dir (str): The directory where plugin-related metadata is stored.

    Methods:
    - __call__(benchmark: List[str]) -> List[str]: Processes multiple benchmarks and returns unique YAML paths.
    - find_registered_benchmarks(root_folder: str) -> List[str]: Finds registered benchmarks in `__init__.py` files.
    - load_benchmark(identifier: str) -> Optional[object]: Loads a benchmark using `brainscore_vision`.
    - detect_benchmark_architecture(benchmark: nn.Module, benchmark_name: str) -> str: Identifies the benchmark's architecture.
    - get_huggingface_link(benchmark_name: str) -> Optional[str]: Checks if a Hugging Face repository exists for the benchmark.
    - get_benchmark_family(benchmark_name: str) -> Optional[str]: Extracts the benchmark family from the name.
    - create_yaml(benchmark: Any, benchmark_name: str) -> Optional[str]: Generates a YAML metadata file.
    - process_single_benchmark(benchmark_name: str) -> Optional[str]: Processes a single benchmark.

    Notes:
    - Designed for benchmarks following the `brainscore_vision` plugin structure.
    - Uses regex to classify benchmarks based on naming conventions.
    - Handles metadata extraction, storage, and linking to external sources.
    - All error handling is printed to prevent silent failures.
    """

    def __init__(self, benchmark_dir: str, benchmark_type: str = None):
        """
        - Initializes the metadata generator with a specified benchmark directory.

        :param benchmark_dir: str, the directory where benchmark metadata files are stored.
        """
        self.benchmark_dir = benchmark_dir
        self.benchmark_type = benchmark_type 

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

    def find_registered_benchmarks(self, root_folder: str) -> List[str]:
        """
        - Finds all registered benchmarks inside `__init__.py` files within a given root directory.

        :param root_folder: str, the root directory to search for benchmark registrations.
        :return: List[str], a list of benchmark names found in `benchmark_registry` assignments.

        Notes:
        - Recursively searches for `__init__.py` files in the specified directory.
        - Extracts benchmark names assigned to `benchmark_registry[...]` using regex.
        - Logs an error message if any `__init__.py` file cannot be read.
        """

        registered_benchmarks = []
        init_file_path = os.path.join(root_folder, "__init__.py")

        # Ensure that `root_folder` is a benchmark directory (must contain `__init__.py`)
        if not os.path.isfile(init_file_path):
            print(f"ERROR: {root_folder} does not contain an `__init__.py` file.", file=sys.stderr)
            return []
        try:
            with open(init_file_path, "r", encoding="utf-8") as file:
                content = file.read()
            matches = re.findall(r'benchmark_registry\[\s*["\'](.*?)["\']\s*\]\s*=', content, re.DOTALL)
            if matches:
                registered_benchmarks.extend(matches)
        except Exception as e:
            print(f"ERROR: Could not read {init_file_path}: {e}", file=sys.stderr)
        return registered_benchmarks

    def load_benchmark(self, identifier: str) -> Optional[object]:
        """
        - Loads a benchmark using `brainscore_vision` and returns the benchmark instance.

        :param identifier: str, the unique name of the benchmark to load.
        :return: Optional[object], the benchmark instance if successfully loaded, otherwise None.

        Notes:
        - Uses `import_plugin` to dynamically load the benchmark from `brainscore_vision.benchmark`.
        - Retrieves the benchmark instance from `benchmark_registry` using the given identifier.
        - Returns `None` if an error occurs during benchmark loading.
        - Prints an error message if the benchmark fails to load.
        """

        try:
            import_plugin('brainscore_vision', 'benchmarks', identifier)
            from brainscore_vision import benchmark_registry
            benchmark_instance = benchmark_registry[identifier]()
            return benchmark_instance
        except Exception as e:
            error_message = f"ERROR: Failed to load benchmark '{identifier}': {e}"
            print(error_message, file=sys.stderr)
            return None, None

    def create_stimuli_metadata(self, plugin, plugin_dir_name):

        def get_num_stimuli(stimulus_set):
            try:
                num_stimuli = len(stimulus_set)
                return num_stimuli
            except TypeError:
                return None

        def total_size_mb(stimulus_set):
            try:
                size = round(float(stimulus_set.memory_usage(deep=True).sum() / (1024 ** 2)), 4)
                return size
            except AttributeError:
                return None

        try:
            stimulus_set = plugin._assembly.stimulus_set
        except AttributeError:
            try:
                stimulus_set = plugin.stimulus_set
            except AttributeError:
                stimulus_set = None

        new_metadata = {
            "num_stimuli": get_num_stimuli(stimulus_set),
            "datatype": "image",
            "stimuli_subtype": None,
            "total_size_mb": total_size_mb(stimulus_set),
            "brainscore_link": f"https://github.com/brain-score/vision/tree/master/brainscore_vision/data/{plugin_dir_name}",
            "extra_notes": None
        }
        return new_metadata

    def create_data_metadata(self, benchmark, benchmark_dir_name):
        try:
            assembly = benchmark._assembly
        except AttributeError:
            try:
                assembly = benchmark.assembly
            except AttributeError:
                assembly = None

        def get_hemisphere(assembly):
            try:
                hemisphere = list(set(assembly.hemisphere.values))[0]
                return hemisphere
            except AttributeError:
                return None

        def get_num_subjects(assembly):
            try:
                num_subjects = len(set(assembly.subject.values))
                return num_subjects
            except AttributeError:
                return None

        def get_region(assembly):
            try:
                region = list(set(assembly.region.values))[0]
                return region
            except AttributeError:
                return None

        def get_datatype():
            if self.benchmark_type == "engineering":
                return "engineering"
            elif self.benchmark_type == "behavioral":
                return "behavioral"
            else:  # either neural or unspecified will return None
                return None

        new_metadata = {
            "benchmark_type": self.benchmark_type,
            "task": None,
            "region": get_region(assembly),
            "hemisphere": get_hemisphere(assembly),
            "num_recording_sites": None,
            "duration_ms": None,
            "species": None,
            "datatype": get_datatype(),
            "num_subjects": get_num_subjects(assembly),
            "pre_processing": None,
            "brainscore_link": f"https://github.com/brain-score/vision/tree/master/brainscore_vision/data/{benchmark_dir_name}",
            "extra_notes": None
        }

        return new_metadata

    def create_metric_metadata(self, plugin, plugin_dir_name):

        new_metadata = {
            "type": None,
            "reference": None,
            "public": None,
            "brainscore_link": f"https://github.com/brain-score/vision/tree/master/brainscore_vision/benchmarks/{plugin_dir_name}",
            "extra_notes": None
        }

        return new_metadata

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
                "stimulus_set": self.create_stimuli_metadata(benchmark, benchmark_dir_name),
                "data": self.create_data_metadata(benchmark, benchmark_dir_name),
                "metric": self.create_metric_metadata(benchmark, benchmark_dir_name),
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
            benchmark= self.load_benchmark(benchmark_name)
            if benchmark is None:
                return
            yaml_path = self.create_yaml(benchmark, benchmark_name, self.benchmark_dir)
            return yaml_path
        except Exception as e:
            error_message = f"ERROR: Unexpected error processing '{benchmark_name}': {e}"
            print(error_message, file=sys.stderr)
            return None
