import os
import sys
import yaml
import argparse
import importlib
from collections import defaultdict, Counter
from typing import Dict, List, Any, Set
from brainscore_core.plugin_management.generate_benchmark_metadata import BenchmarkMetadataGenerator


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


class TemplateExtractor:
    """
    Extracts templates from expanded benchmark metadata by analyzing patterns.
    
    This tool analyzes repetitive benchmark metadata and automatically generates:
    1. Data plugin templates (common stimulus_set and data configurations)
    2. Metric plugin templates (common metric configurations)
    3. Inheritance-format benchmark metadata (data_id/metric_id references)
    """
    
    def __init__(self, domain_plugin):
        self.domain_plugin = domain_plugin
    
    def extract_templates_from_benchmarks(self, benchmark_list: List[str], benchmark_dir: str) -> Dict[str, Any]:
        """
        Analyze benchmarks and extract common patterns for template generation.
        
        :param benchmark_list: List of benchmark names to analyze
        :param benchmark_dir: Directory containing the benchmarks
        :return: Dictionary containing extracted templates and inheritance metadata
        """
        print(f"Analyzing {len(benchmark_list)} benchmarks for template extraction...", file=sys.stderr)
        
        # Step 1: Generate expanded metadata for all benchmarks
        expanded_metadata = {}
        data_ids = {}
        metric_ids = {}
        benchmark_dir_name = benchmark_dir.split("/")[-1]
        
        for i, benchmark_name in enumerate(benchmark_list):
            print(f"Analyzing benchmark {i + 1}/{len(benchmark_list)}: {benchmark_name}", file=sys.stderr)
            try:
                print(f"  Loading benchmark: {benchmark_name}", file=sys.stderr)
                benchmark = self.domain_plugin.load_benchmark(benchmark_name)
                if benchmark is None:
                    print(f"  Benchmark {benchmark_name} returned None, skipping", file=sys.stderr)
                    continue
                
                print(f"  Extracting metadata for: {benchmark_name}", file=sys.stderr)
                
                # Generate expanded metadata using raw extraction (bypass templates)
                try:
                    stimulus_meta = self._extract_raw_stimuli_metadata(benchmark, benchmark_dir_name)
                    data_meta = self._extract_raw_data_metadata(benchmark, benchmark_dir_name)
                    metric_meta = self._extract_raw_metric_metadata(benchmark, benchmark_dir_name)
                    
                    metadata = {
                        "stimulus_set": stimulus_meta,
                        "data": data_meta,
                        "metric": metric_meta,
                    }
                    expanded_metadata[benchmark_name] = metadata
                    print(f"  Successfully extracted metadata for: {benchmark_name}", file=sys.stderr)
                    
                except Exception as extraction_error:
                    print(f"  ERROR during metadata extraction for '{benchmark_name}': {extraction_error}", file=sys.stderr)
                    print(f"  Skipping metadata extraction but continuing with ID extraction...", file=sys.stderr)
                
                # Extract IDs for inheritance format (even if metadata extraction failed)
                try:
                    if hasattr(self.domain_plugin, '_extract_data_id'):
                        data_id = self.domain_plugin._extract_data_id(benchmark)
                        if data_id:
                            data_ids[benchmark_name] = data_id
                            print(f"  Extracted data_id: {data_id}", file=sys.stderr)
                    
                    if hasattr(self.domain_plugin, '_extract_metric_id'):
                        metric_id = self.domain_plugin._extract_metric_id(benchmark)
                        if metric_id:
                            metric_ids[benchmark_name] = metric_id
                            print(f"  Extracted metric_id: {metric_id}", file=sys.stderr)
                except Exception as id_error:
                    print(f"  ERROR during ID extraction for '{benchmark_name}': {id_error}", file=sys.stderr)
                        
            except Exception as e:
                print(f"ERROR: Failed to load benchmark '{benchmark_name}': {e}", file=sys.stderr)
                print(f"  This benchmark has loading issues, skipping entirely...", file=sys.stderr)
                continue
        
        # Step 2: Analyze patterns and extract templates
        templates = self._analyze_patterns(expanded_metadata, data_ids, metric_ids)
        
        return templates
    
    def _analyze_patterns(self, expanded_metadata: Dict[str, Dict], data_ids: Dict[str, str], metric_ids: Dict[str, str]) -> Dict[str, Any]:
        """Analyze expanded metadata to find common patterns."""
        
        # Group by data plugin and metric plugin
        data_plugin_groups = defaultdict(list)
        metric_plugin_groups = defaultdict(list)
        
        for benchmark_name, metadata in expanded_metadata.items():
            data_id = data_ids.get(benchmark_name)
            metric_id = metric_ids.get(benchmark_name)
            
            if data_id and hasattr(self.domain_plugin, '_extract_data_plugin_name'):
                data_plugin = self.domain_plugin._extract_data_plugin_name(data_id)
                if data_plugin:
                    data_plugin_groups[data_plugin].append({
                        'benchmark_name': benchmark_name,
                        'data_id': data_id,
                        'stimulus_set': metadata['stimulus_set'],
                        'data': metadata['data']
                    })
            
            if metric_id:
                metric_plugin_groups[metric_id].append({
                    'benchmark_name': benchmark_name,
                    'metric': metadata['metric']
                })
        
        # Extract data templates
        data_templates = {}
        for data_plugin, benchmarks in data_plugin_groups.items():
            data_templates[data_plugin] = self._extract_data_template(benchmarks)
        
        # Extract metric templates  
        metric_templates = {}
        for metric_plugin, benchmarks in metric_plugin_groups.items():
            metric_templates[metric_plugin] = self._extract_metric_template(benchmarks)
        
        # Generate inheritance format
        inheritance_metadata = {}
        for benchmark_name in expanded_metadata.keys():
            data_id = data_ids.get(benchmark_name)
            metric_id = metric_ids.get(benchmark_name)
            if data_id and metric_id:
                inheritance_metadata[benchmark_name] = {
                    'data_id': data_id,
                    'metric_id': metric_id
                }
        
        return {
            'data_templates': data_templates,
            'metric_templates': metric_templates,
            'inheritance_metadata': inheritance_metadata,
            'expanded_metadata': expanded_metadata  # For comparison
        }
    
    def _extract_data_template(self, benchmarks: List[Dict]) -> Dict[str, Any]:
        """Extract common data template from benchmarks."""
        if not benchmarks:
            return {}
        
        # Find common fields in stimulus_set
        stimulus_common = self._find_common_fields([b['stimulus_set'] for b in benchmarks])
        
        # Find common fields in data
        data_common = self._find_common_fields([b['data'] for b in benchmarks])
        
        # Find dataset-specific differences
        data_specific = {}
        for benchmark in benchmarks:
            data_id = benchmark['data_id']
            
            # Find stimulus_set differences
            stimulus_diff = self._find_differences(benchmark['stimulus_set'], stimulus_common)
            
            # Find data differences  
            data_diff = self._find_differences(benchmark['data'], data_common)
            
            if stimulus_diff or data_diff:
                data_specific[data_id] = {}
                if stimulus_diff:
                    data_specific[data_id]['stimulus_set'] = stimulus_diff
                if data_diff:
                    data_specific[data_id]['data_assembly'] = data_diff
        
        return {
            'defaults': {
                'stimulus_set': stimulus_common,
                'data_assembly': data_common
            },
            'data': data_specific
        }
    
    def _extract_metric_template(self, benchmarks: List[Dict]) -> Dict[str, Any]:
        """Extract common metric template from benchmarks."""
        if not benchmarks:
            return {}
        
        # All metric configs should be identical, so take the first one
        return benchmarks[0]['metric']
    
    def _find_common_fields(self, field_list: List[Dict]) -> Dict[str, Any]:
        """Find fields that have the same value across all dictionaries."""
        if not field_list:
            return {}
        
        common = {}
        first_dict = field_list[0]
        
        for key, value in first_dict.items():
            # Check if this key-value pair exists in all dictionaries
            if all(d.get(key) == value for d in field_list):
                common[key] = value
        
        return common
    
    def _find_differences(self, item_dict: Dict, common_dict: Dict) -> Dict[str, Any]:
        """Find fields that differ from the common template."""
        differences = {}
        
        for key, value in item_dict.items():
            if key not in common_dict or common_dict[key] != value:
                differences[key] = value
        
        return differences
    
    def _extract_raw_stimuli_metadata(self, plugin, plugin_dir_name: str) -> Dict[str, Any]:
        """Extract stimuli metadata using raw extraction (bypass templates)."""
        def get_num_stimuli(stimulus_set):
            try:
                if stimulus_set is None:
                    return None
                return len(stimulus_set)
            except (TypeError, AttributeError, ValueError):
                return None

        def total_size_mb(stimulus_set):
            try:
                if stimulus_set is None:
                    return None
                return round(float(stimulus_set.memory_usage(deep=True).sum() / (1024 ** 2)), 4)
            except (AttributeError, TypeError, ValueError):
                return None

        stimulus_set = None
        try:
            # Try multiple ways to get stimulus_set
            if hasattr(plugin, '_assembly') and plugin._assembly is not None:
                stimulus_set = getattr(plugin._assembly, 'stimulus_set', None)
            if stimulus_set is None and hasattr(plugin, 'stimulus_set'):
                stimulus_set = plugin.stimulus_set
            if stimulus_set is None and hasattr(plugin, '_stimulus_set'):
                stimulus_set = plugin._stimulus_set
        except (AttributeError, TypeError):
            stimulus_set = None

        return {
            "num_stimuli": get_num_stimuli(stimulus_set),
            "datatype": "image",
            "stimuli_subtype": None,
            "total_size_mb": total_size_mb(stimulus_set),
            "brainscore_link": f"https://github.com/brain-score/vision/tree/master/brainscore_vision/data/{plugin_dir_name}",
            "extra_notes": None
        }
    
    def _extract_raw_data_metadata(self, plugin, plugin_dir_name: str) -> Dict[str, Any]:
        """Extract data metadata using raw extraction (bypass templates)."""
        assembly = None
        try:
            if hasattr(plugin, '_assembly'):
                assembly = plugin._assembly
            elif hasattr(plugin, 'assembly'):
                assembly = plugin.assembly
        except (AttributeError, TypeError):
            assembly = None

        def get_hemisphere(assembly):
            try:
                if assembly is None:
                    return None
                hemisphere_values = set(assembly.hemisphere.values)
                return "L" if "L" in hemisphere_values else sorted(list(hemisphere_values))[0] if hemisphere_values else None
            except (AttributeError, TypeError, ValueError):
                return None

        def get_num_subjects(assembly):
            try:
                if assembly is None:
                    return None
                return len(set(assembly.subject.values))
            except (AttributeError, TypeError, ValueError):
                return None

        def get_region(assembly):
            try:
                if assembly is None:
                    return None
                # Handle both single values and arrays safely
                region_values = assembly.region.values
                if hasattr(region_values, '__len__') and len(region_values) > 0:
                    import numpy as np
                    unique_regions = np.unique(region_values)
                    # Convert numpy types to Python native types
                    return str(unique_regions[0]) if len(unique_regions) > 0 else None
                else:
                    return str(region_values) if region_values is not None else None
            except (AttributeError, TypeError, ValueError, IndexError):
                # Fallback: try to get region from plugin directly
                try:
                    region = getattr(plugin, 'region', None)
                    return str(region) if region is not None else None
                except:
                    return None

        # Determine benchmark type from plugin
        benchmark_type = "neural"  # default
        try:
            if hasattr(plugin, 'parent') and plugin.parent:
                parent = str(plugin.parent).lower()
                if 'behavior' in parent:
                    benchmark_type = "behavioral"
                elif 'engineering' in parent:
                    benchmark_type = "engineering"
                elif 'neural' in parent:
                    benchmark_type = "neural"
        except (AttributeError, TypeError):
            pass

        def get_datatype():
            if benchmark_type == "engineering":
                return "engineering"
            elif benchmark_type == "behavioral":
                return "behavioral"
            else:
                return None

        return {
            "benchmark_type": benchmark_type,
            "task": None,
            "region": get_region(assembly),
            "hemisphere": get_hemisphere(assembly),
            "num_recording_sites": None,
            "duration_ms": None,
            "species": None,
            "datatype": get_datatype(),
            "num_subjects": get_num_subjects(assembly),
            "pre_processing": None,
            "brainscore_link": f"https://github.com/brain-score/vision/tree/master/brainscore_vision/data/{plugin_dir_name}",
            "extra_notes": None,
            "data_publicly_available": True
        }
    
    def _extract_raw_metric_metadata(self, plugin, plugin_dir_name: str) -> Dict[str, Any]:
        """Extract metric metadata using raw extraction (bypass templates)."""
        metric_type = None
        
        # Try to determine metric type from plugin
        if hasattr(plugin, '_metric') and hasattr(plugin._metric, '__class__'):
            metric_class_name = plugin._metric.__class__.__name__
            if 'Accuracy' in metric_class_name:
                metric_type = 'accuracy'
            elif 'ErrorConsistency' in metric_class_name:
                metric_type = 'error_consistency'
            elif 'ValueDelta' in metric_class_name:
                metric_type = 'value_delta'
            elif 'PLS' in metric_class_name or 'Regression' in metric_class_name:
                metric_type = 'pls'
        
        # Also check benchmark identifier
        if not metric_type and hasattr(plugin, 'identifier'):
            identifier = plugin.identifier.lower()
            if 'top1' in identifier or 'accuracy' in identifier:
                metric_type = 'accuracy'
            elif 'error_consistency' in identifier:
                metric_type = 'error_consistency'
            elif 'pls' in identifier:
                metric_type = 'pls'
            elif 'value_delta' in identifier:
                metric_type = 'value_delta'
            elif 'rdm' in identifier:
                metric_type = 'rdm'
            elif 'coggan' in identifier and 'fmri' in identifier:
                metric_type = 'rdm'

        return {
            "type": metric_type,
            "reference": None,
            "public": None,
            "brainscore_link": f"https://github.com/brain-score/vision/tree/master/brainscore_vision/metrics/{metric_type}" if metric_type else f"https://github.com/brain-score/vision/tree/master/brainscore_vision/benchmarks/{plugin_dir_name}",
            "extra_notes": None
        }
    
    def write_templates(self, templates: Dict[str, Any], benchmark_dir: str):
        """Write extracted templates to appropriate plugin directories."""
        
        # Write data templates
        for data_plugin, template in templates['data_templates'].items():
            self._write_data_template(data_plugin, template)
        
        # Write metric templates
        for metric_plugin, template in templates['metric_templates'].items():
            self._write_metric_template(metric_plugin, template)
        
        # Write inheritance format benchmark metadata
        inheritance_metadata = templates['inheritance_metadata']
        if inheritance_metadata:
            self._write_inheritance_metadata(inheritance_metadata, benchmark_dir)
    
    def _write_data_template(self, data_plugin: str, template: Dict[str, Any]):
        """Write data template to data plugin directory."""
        try:
            # Find brainscore_vision directory relative to current working directory
            vision_dir = None
            current_dir = os.getcwd()
            
            # Look for vision/brainscore_vision in current directory
            if os.path.exists(os.path.join(current_dir, 'vision', 'brainscore_vision')):
                vision_dir = os.path.join(current_dir, 'vision', 'brainscore_vision')
            elif os.path.exists(os.path.join(current_dir, 'brainscore_vision')):
                vision_dir = os.path.join(current_dir, 'brainscore_vision')
            else:
                print(f"ERROR: Could not find brainscore_vision directory from {current_dir}", file=sys.stderr)
                return
            
            data_dir = os.path.join(vision_dir, 'data', data_plugin)
            if not os.path.exists(data_dir):
                print(f"Warning: Data plugin directory {data_dir} does not exist", file=sys.stderr)
                return
            
            template_path = os.path.join(data_dir, 'metadata.yaml')
            
            with open(template_path, 'w') as f:
                yaml.dump(template, f, default_flow_style=False, sort_keys=False, indent=2)
            
            print(f"Created data template: {template_path}", file=sys.stderr)
            
        except Exception as e:
            print(f"ERROR: Failed to write data template for {data_plugin}: {e}", file=sys.stderr)
    
    def _write_metric_template(self, metric_plugin: str, template: Dict[str, Any]):
        """Write metric template to metric plugin directory."""
        try:
            # Find brainscore_vision directory relative to current working directory
            vision_dir = None
            current_dir = os.getcwd()
            
            # Look for vision/brainscore_vision in current directory
            if os.path.exists(os.path.join(current_dir, 'vision', 'brainscore_vision')):
                vision_dir = os.path.join(current_dir, 'vision', 'brainscore_vision')
            elif os.path.exists(os.path.join(current_dir, 'brainscore_vision')):
                vision_dir = os.path.join(current_dir, 'brainscore_vision')
            else:
                print(f"ERROR: Could not find brainscore_vision directory from {current_dir}", file=sys.stderr)
                return
            
            metric_dir = os.path.join(vision_dir, 'metrics', metric_plugin)
            if not os.path.exists(metric_dir):
                print(f"Warning: Metric plugin directory {metric_dir} does not exist", file=sys.stderr)
                return
            
            template_path = os.path.join(metric_dir, 'metadata.yaml')
            
            with open(template_path, 'w') as f:
                yaml.dump(template, f, default_flow_style=False, sort_keys=False, indent=2)
            
            print(f"Created metric template: {template_path}", file=sys.stderr)
            
        except Exception as e:
            print(f"ERROR: Failed to write metric template for {metric_plugin}: {e}", file=sys.stderr)
    
    def _write_inheritance_metadata(self, inheritance_metadata: Dict[str, Any], benchmark_dir: str):
        """Write inheritance format benchmark metadata."""
        try:
            yaml_path = os.path.join(benchmark_dir, 'metadata.yaml')
            
            final_metadata = {"benchmarks": inheritance_metadata}
            
            with open(yaml_path, 'w') as f:
                yaml.dump(final_metadata, f, default_flow_style=False, sort_keys=False, indent=4)
            
            print(f"Created inheritance format metadata: {yaml_path}", file=sys.stderr)
            
        except Exception as e:
            print(f"ERROR: Failed to write inheritance metadata: {e}", file=sys.stderr)


def main():
    """Command-line interface for template extraction."""
    parser = argparse.ArgumentParser(description="Extract templates from repetitive benchmark metadata")
    parser.add_argument("--plugin-dir", required=True, help="Path to the benchmark plugin directory")
    parser.add_argument("--domain", default="vision", help="Domain type")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be extracted without writing files")
    
    args = parser.parse_args()
    
    # Load domain plugin
    domain_plugin = load_domain_plugin(args.domain)
    if domain_plugin is None:
        print(f"ERROR: Could not load domain plugin for '{args.domain}'", file=sys.stderr)
        return
    
    # Create extractor
    extractor = TemplateExtractor(domain_plugin)
    
    # Find registered benchmarks
    benchmark_list = domain_plugin.find_registered_benchmarks(args.plugin_dir)
    if not benchmark_list:
        print("ERROR: No benchmarks found", file=sys.stderr)
        return
    
    # Extract templates
    templates = extractor.extract_templates_from_benchmarks(benchmark_list, args.plugin_dir)
    
    if args.dry_run:
        print("=== DRY RUN - Would create these templates ===", file=sys.stderr)
        print(f"Data templates: {list(templates['data_templates'].keys())}", file=sys.stderr)
        print(f"Metric templates: {list(templates['metric_templates'].keys())}", file=sys.stderr)
        print(f"Inheritance metadata: {len(templates['inheritance_metadata'])} benchmarks", file=sys.stderr)
        
        # Show sample data template
        if templates['data_templates']:
            data_plugin = list(templates['data_templates'].keys())[0]
            print(f"\nSample data template for {data_plugin}:", file=sys.stderr)
            print(yaml.dump(templates['data_templates'][data_plugin], default_flow_style=False, indent=2), file=sys.stderr)
    else:
        # Write templates
        extractor.write_templates(templates, args.plugin_dir)
        print("Template extraction completed!", file=sys.stderr)


if __name__ == "__main__":
    main()
