import os
import sys
import yaml
import argparse
from collections import defaultdict
from typing import Dict, List, Any


class ExistingMetadataExtractor:
    """
    Extract templates from existing metadata.yaml files without loading benchmarks.
    
    This is useful when benchmarks have loading issues but the metadata is already correct.
    """
    
    def extract_from_file(self, metadata_file_path: str) -> Dict[str, Any]:
        """
        Extract templates from an existing metadata file by analyzing patterns.
        
        :param metadata_file_path: Path to existing metadata.yaml file
        :return: Dictionary containing extracted templates
        """
        try:
            with open(metadata_file_path, 'r') as f:
                metadata = yaml.safe_load(f)
        except Exception as e:
            print(f"ERROR: Could not read {metadata_file_path}: {e}", file=sys.stderr)
            return {}
        
        if 'benchmarks' not in metadata:
            print("ERROR: No 'benchmarks' section found in metadata", file=sys.stderr)
            return {}
        
        benchmarks = metadata['benchmarks']
        print(f"Analyzing {len(benchmarks)} benchmarks from existing metadata...", file=sys.stderr)
        
        return self._analyze_existing_patterns(benchmarks)
    
    def _analyze_existing_patterns(self, benchmarks: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze existing benchmark metadata to extract patterns."""
        
        # Group benchmarks by potential data/metric plugins
        data_groups = defaultdict(list)
        metric_groups = defaultdict(list)
        
        for benchmark_name, benchmark_meta in benchmarks.items():
            # Extract potential data plugin name from benchmark name
            data_plugin = self._guess_data_plugin(benchmark_name)
            
            # Extract potential metric type from benchmark metadata
            metric_type = benchmark_meta.get('metric', {}).get('type', 'unknown')
            
            data_groups[data_plugin].append({
                'name': benchmark_name,
                'stimulus_set': benchmark_meta.get('stimulus_set', {}),
                'data': benchmark_meta.get('data', {}),
                'inferred_data_id': self._guess_data_id(benchmark_name)
            })
            
            metric_groups[metric_type].append({
                'name': benchmark_name,
                'metric': benchmark_meta.get('metric', {})
            })
        
        # Extract templates
        data_templates = {}
        for data_plugin, benchmarks_list in data_groups.items():
            if len(benchmarks_list) > 1:  # Only extract if there's repetition
                data_templates[data_plugin] = self._extract_data_template_from_existing(benchmarks_list)
        
        metric_templates = {}
        for metric_type, benchmarks_list in metric_groups.items():
            if len(benchmarks_list) > 1 and metric_type != 'unknown':  # Only extract if there's repetition
                metric_templates[metric_type] = self._extract_metric_template_from_existing(benchmarks_list)
        
        # Generate inheritance format
        inheritance_metadata = {}
        for benchmark_name, benchmark_meta in benchmarks.items():
            data_plugin = self._guess_data_plugin(benchmark_name)
            metric_type = benchmark_meta.get('metric', {}).get('type', None)
            data_id = self._guess_data_id(benchmark_name)
            
            if data_plugin in data_templates and metric_type in metric_templates:
                inheritance_metadata[benchmark_name] = {
                    'data_id': data_id,
                    'metric_id': metric_type
                }
        
        return {
            'data_templates': data_templates,
            'metric_templates': metric_templates,
            'inheritance_metadata': inheritance_metadata,
            'original_metadata': benchmarks
        }
    
    def _guess_data_plugin(self, benchmark_name: str) -> str:
        """Guess data plugin name from benchmark name."""
        # Handle common patterns
        if 'Coggan2024_fMRI' in benchmark_name:
            return 'coggan2024_fmri'
        elif 'Coggan2024_behavior' in benchmark_name:
            return 'coggan2024_behavior'
        elif 'Ferguson2024' in benchmark_name:
            return 'ferguson2024'
        elif 'MajajHong2015' in benchmark_name:
            return 'majajhong2015'
        elif 'Geirhos2021' in benchmark_name:
            return 'geirhos2021'
        else:
            # Extract from first part before dot or dash
            parts = benchmark_name.replace('.', '-').split('-')
            return parts[0].lower() if parts else benchmark_name.lower()
    
    def _guess_data_id(self, benchmark_name: str) -> str:
        """Guess data ID from benchmark name."""
        # Handle common patterns
        if 'tong.Coggan2024_fMRI' in benchmark_name:
            # tong.Coggan2024_fMRI.V1-rdm -> Coggan2024_fMRI_V1
            parts = benchmark_name.split('.')
            if len(parts) >= 3:
                region_part = parts[2].split('-')[0]  # V1 from V1-rdm
                return f"Coggan2024_fMRI_{region_part}"
        elif 'Coggan2024_behavior' in benchmark_name:
            return 'Coggan2024_behavior'
        elif 'Ferguson2024' in benchmark_name:
            # Ferguson2024circle_line-value_delta -> Ferguson2024_circle_line
            return benchmark_name.split('-')[0].replace('Ferguson2024', 'Ferguson2024_')
        elif 'Geirhos2021' in benchmark_name:
            # Geirhos2021colour-top1 -> Geirhos2021_colour
            return benchmark_name.split('-')[0].replace('Geirhos2021', 'Geirhos2021_')
        
        return benchmark_name
    
    def _extract_data_template_from_existing(self, benchmarks_list: List[Dict]) -> Dict[str, Any]:
        """Extract data template from existing benchmark metadata."""
        if not benchmarks_list:
            return {}
        
        # Find common fields
        stimulus_common = self._find_common_fields([b['stimulus_set'] for b in benchmarks_list])
        data_common = self._find_common_fields([b['data'] for b in benchmarks_list])
        
        # Find differences for each benchmark
        data_specific = {}
        for benchmark in benchmarks_list:
            data_id = benchmark['inferred_data_id']
            
            stimulus_diff = self._find_differences(benchmark['stimulus_set'], stimulus_common)
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
    
    def _extract_metric_template_from_existing(self, benchmarks_list: List[Dict]) -> Dict[str, Any]:
        """Extract metric template from existing benchmark metadata."""
        if not benchmarks_list:
            return {}
        
        # All metric configs should be identical, so take the first one
        return benchmarks_list[0]['metric']
    
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


def main():
    """Command-line interface for extracting from existing metadata."""
    parser = argparse.ArgumentParser(description="Extract templates from existing metadata files")
    parser.add_argument("--metadata-file", required=True, help="Path to existing metadata.yaml file")
    parser.add_argument("--output-dir", help="Directory to write templates to (default: infer from metadata file)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be extracted without writing files")
    
    args = parser.parse_args()
    
    extractor = ExistingMetadataExtractor()
    templates = extractor.extract_from_file(args.metadata_file)
    
    if args.dry_run:
        print("=== EXTRACTED FROM EXISTING METADATA ===", file=sys.stderr)
        print(f"Data templates: {list(templates['data_templates'].keys())}", file=sys.stderr)
        print(f"Metric templates: {list(templates['metric_templates'].keys())}", file=sys.stderr)
        print(f"Inheritance metadata: {len(templates['inheritance_metadata'])} benchmarks", file=sys.stderr)
    else:
        print("Template extraction from existing metadata completed!", file=sys.stderr)


if __name__ == "__main__":
    main()

