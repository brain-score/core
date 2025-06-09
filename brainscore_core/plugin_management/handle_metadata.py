import os
import sys
import yaml
import json
import argparse
import subprocess
import time
import importlib

from brainscore_core.submission.endpoints import MetadataEndpoint
from brainscore_core.plugin_management.generate_model_metadata import ModelMetadataGenerator
from brainscore_core.plugin_management.generate_benchmark_metadata import BenchmarkMetadataGenerator

# allowed plugin types for metadata
ALLOWED_PLUGINS = {
    "models",
    "benchmarks"
}

# allowed keys for plugin metadata files
ALLOWED_KEYS_BY_TYPE = {
    "models": {
        "architecture",
        "model_family",
        "total_parameter_count",
        "trainable_parameter_count",
        "total_layers",
        "trainable_layers",
        "model_size_mb",
        "training_dataset",
        "task_specialization",
        "brainscore_link",
        "huggingface_link",
        "extra_notes",
        "runnable"
    },
    "benchmarks": {
        "stimulus_set": {
            "num_stimuli",
            "datatype",
            "stimuli_subtype",
            "total_size_mb",
            "brainscore_link",
            "extra_notes"
        },
        "data": {
            "benchmark_type",
            "task",
            "region",
            "hemisphere",
            "num_recording_sites",
            "duration_ms",
            "species",
            "datatype",
            "num_subjects",
            "pre_processing",
            "brainscore_link",
            "extra_notes",
            "data_publicly_available"
        },
        "metric": {
            "type",
            "reference",
            "public",
            "brainscore_link",
            "extra_notes"
        }
    }
}


def validate_metadata_file(metadata_path):
    """
    Validate a metadata file that is expected to have one of the allowed plugin types
    as its top-level key (e.g. "models" or "benchmarks"). For plugin types that have an
    allowed keys set defined, check that each plugin's metadata does not include extra keys.

    Returns a tuple of (errors, data) where errors is a list of strings and data is the loaded YAML.
    """
    errors = []
    try:
        with open(metadata_path, 'r') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return [f"YAML parsing error: {e}"], None
    except Exception as e:
        return [f"Error reading file: {e}"], None

    if not isinstance(data, dict):
        errors.append("Top-level structure must be a dictionary.")
        return errors, None

    # check that at least one allowed plugin type is present.
    found_plugin_type = False
    for plugin_type in data:
        if plugin_type in ALLOWED_PLUGINS:
            found_plugin_type = True
            plugin_data = data[plugin_type]
            if not isinstance(plugin_data, dict):
                errors.append(f"'{plugin_type}' value must be a dictionary keyed by plugin names.")
                continue

            # find keys based on plugin type
            allowed_keys = ALLOWED_KEYS_BY_TYPE.get(plugin_type)
            for plugin_name, metadata in plugin_data.items():
                if not isinstance(metadata, dict):
                    errors.append(f"Data for plugin '{plugin_name}' under '{plugin_type}' must be a dictionary.")
                    continue
                if plugin_type == "benchmarks":  # for benchmarks check two layers of keys
                    for top_key, top_value in metadata.items():
                        if top_key not in allowed_keys:
                            errors.append(
                                f"Plugin '{plugin_name}' has invalid top-level key: '{top_key}'. Allowed keys: {list(allowed_keys.keys())}"
                            )
                            continue

                        if not isinstance(top_value, dict):
                            errors.append(f"Value for '{top_key}' in plugin '{plugin_name}' must be a dictionary.")
                            continue

                        # Check nested keys
                        nested_allowed_keys = allowed_keys[top_key]
                        extra_nested_keys = set(top_value.keys()) - nested_allowed_keys
                        if extra_nested_keys:
                            errors.append(
                                f"Plugin '{plugin_name}' has extra keys under '{top_key}': {list(extra_nested_keys)}"
                            )
                else:
                    extra_keys = set(metadata.keys()) - allowed_keys
                    if extra_keys:
                        errors.append(
                            f"Plugin '{plugin_name}' under '{plugin_type}' has extra keys: {list(extra_keys)}"
                        )
        else:
            errors.append(f"Top-level key '{plugin_type}' is not allowed. Allowed keys: {list(ALLOWED_PLUGINS)}")

    if not found_plugin_type:
        errors.append(f"Missing one of the required top-level keys: {list(ALLOWED_PLUGINS)}")

    return errors, data


def load_domain_plugin(domain: str, benchmark_type: str = "neural"):
    """
    Dynamically load domain plugin without circular dependencies.
    
    Uses importlib to dynamically import domain plugins following naming convention:
    brainscore_{domain}.plugin_management.{Domain}DomainPlugin
    
    :param domain: str, the domain name (e.g., "vision", "audio", "language")
    :param benchmark_type: str, the benchmark type for domain plugin initialization (optional for models)
    :return: Domain plugin instance
    """
    try:
        # Dynamic import following naming convention
        domain_module_name = f"brainscore_{domain}.plugin_management"
        domain_class_name = f"{domain.capitalize()}DomainPlugin"
        
        # Import the module
        domain_module = importlib.import_module(domain_module_name)
        
        # Get the plugin class
        domain_plugin_class = getattr(domain_module, domain_class_name)
        
        # Instantiate the plugin
        domain_plugin = domain_plugin_class(benchmark_type=benchmark_type)
        
        return domain_plugin
        
    except ImportError as e:
        print(f"ERROR: Could not import {domain_module_name}: {e}", file=sys.stderr)
        print(f"Please install brainscore_{domain} package.", file=sys.stderr)
        return None
    except AttributeError as e:
        print(f"ERROR: Could not find {domain_class_name} in {domain_module_name}: {e}", file=sys.stderr)
        return None


def generate_metadata(plugin_dir, plugin_type, benchmark_type="neural", domain="vision"):
    """Generate metadata using domain plugins for both models and benchmarks."""
    
    # Use dynamic plugin loading instead of hardcoded imports
    domain_plugin = load_domain_plugin(domain, benchmark_type)
    if domain_plugin is None:
        return None
    
    if plugin_type == "models":
        generator = ModelMetadataGenerator(plugin_dir, domain_plugin)
        model_list = generator.find_registered_models(plugin_dir)
        metadata_path = generator(model_list)
        metadata_path = metadata_path[0] if metadata_path else None
    elif plugin_type == "benchmarks":
        generator = BenchmarkMetadataGenerator(plugin_dir, domain_plugin)
        benchmark_list = generator.find_registered_benchmarks(plugin_dir)
        metadata_path = generator(benchmark_list)
        metadata_path = metadata_path[0] if metadata_path else None
    else:
        raise ValueError(f"Unsupported plugin type: {plugin_type}")

    print(f"{plugin_type} metadata.yml generated at: {metadata_path}", file=sys.stderr)
    return metadata_path


def create_metadata_pr(plugin_dir, branch_name="auto/metadata-update"):
    """
    Create a PR that adds/updates the metadata.yml file.

    This function uses git commands and the GitHub CLI (gh) to:
      - Create a new branch,
      - Add the metadata file,
      - Commit the changes,
      - Push the branch,
      - Create a PR.
    Note: This requires that repo has been checked out with a full history,
    and that gh is installed/authenticated.
    """
    metadata_path = os.path.join(plugin_dir, "metadata.yml")
    unique_suffix = str(int(time.time()))
    branch_name += f"_{unique_suffix}"
    try:
        # create branch, add metadata, commit, push, create PR
        subprocess.run(["git", "checkout", "-b", branch_name], check=True, stdout=sys.stderr, stderr=sys.stderr)
        subprocess.run(["git", "add", metadata_path], check=True, stdout=sys.stderr, stderr=sys.stderr)
        subprocess.run(["git", "commit", "-m", "Auto-add/update metadata.yml for plugin"], check=True, stdout=sys.stderr, stderr=sys.stderr)
        subprocess.run(["git", "push", "origin", branch_name], check=True, stdout=sys.stderr, stderr=sys.stderr)
        pr_title = "Auto-add/update metadata.yml for plugin"
        pr_body = "This PR was automatically generated to add or update the metadata.yml file."
        subprocess.run([
            "gh", "pr", "create",
            "--title", pr_title,
            "--body", pr_body,
            "--label", "automerge-metadata"
        ], check=True, stdout=sys.stderr, stderr=sys.stderr)
        print("Pull request created successfully for metadata.yml update.", file=sys.stderr)
        time.sleep(5)
        pr_number = subprocess.check_output(
            ["gh", "pr", "view", "--json", "number", "--jq", ".number"],
            universal_newlines=True
        ).strip()
        return pr_number
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while creating the PR: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Process and validate metadata.yml for a plugin, "
                    "or generate one and create a PR if missing."
    )
    parser.add_argument("--plugin-dir", required=True, help="Path to the plugin directory")
    parser.add_argument("--plugin-type", required=True, choices=list(ALLOWED_PLUGINS),
                        help="Plugin type (e.g., models or benchmarks)")
    parser.add_argument("--domain", default="vision", choices=["vision"], 
                        help="Domain type (currently only 'vision' is supported)")
    parser.add_argument("--db-connection", action="store_true", default=False,
                        help="If provided, establish a new database connection")
    args = parser.parse_args()

    new_metadata = False
    yml_path = os.path.join(args.plugin_dir, "metadata.yml")
    yaml_path = os.path.join(args.plugin_dir, "metadata.yaml")

    if os.path.exists(yml_path) and os.path.exists(yaml_path):
        print("ERROR: Both metadata.yml and metadata.yaml exist. Please keep only one.", file=sys.stderr)
        sys.exit(1)
    
    # prioritize .yml, fall back to .yaml
    metadata_path = yml_path if os.path.exists(yml_path) else yaml_path if os.path.exists(yaml_path) else None
    
    if metadata_path is None:
        print("No metadata.yml or metadata.yaml found. Generating now.", file=sys.stderr)
        new_metadata = True
        metadata_path = generate_metadata(args.plugin_dir, args.plugin_type, domain=args.domain)
    else:
        print(f"Found metadata at {metadata_path}. Validating...", file=sys.stderr)

    errors, data = validate_metadata_file(metadata_path)
    if errors:
        print("Metadata validation errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)
    else:
        print("metadata.yml is valid.", file=sys.stderr)
        with open("validated_metadata.json", "w") as f:
            json.dump(data, f)
        print("Validated metadata saved to validated_metadata.json", file=sys.stderr)

    if args.db_connection:  # if metadata was altered, must upload to db on new connection
        print("Creating metadata endpoint...", file=sys.stderr)
        db_secret = os.environ.get("BSC_DATABASESECRET")
        
        # Load domain plugin (same as generate_metadata does)
        domain_plugin = load_domain_plugin(args.domain)
        if domain_plugin is None:
            print(f"ERROR: Could not load domain plugin for '{args.domain}'", file=sys.stderr)
            sys.exit(1)
        
        create_endpoint = MetadataEndpoint(domain_plugins=domain_plugin, db_secret=db_secret)
        create_endpoint(plugin_dir=args.plugin_dir, plugin_type=args.plugin_type, domain=args.domain)

    if new_metadata:  # if metadata was created, create a pr that will be automerged and approved by github actions
        pr_number = create_metadata_pr(args.plugin_dir)
        print(pr_number)
    else:
        print("")


if __name__ == "__main__":
    main()
