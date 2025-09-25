"""
SUPPORTED DATA STANDARDS - Data format support for Brain-Score

PURPOSE:
--------
This module contains support for various neuroscience data standards used
by Brain-Score. Each data standard has its own submodule with the necessary
data structures, I/O operations, and utilities.

CURRENT STANDARDS:
-----------------
- brainio: Brain-Score's original data format (assemblies, stimulus sets)
- [future] nwb: Neurodata Without Borders format support

DESIGN PHILOSOPHY:
-----------------
Each data standard module should provide:
- Core data structures (assemblies, stimulus collections)
- I/O operations (load/save from various sources)
- Conversion utilities (between formats if needed)
- Validation and integrity checking

This allows Brain-Score to support multiple data standards while keeping
the implementations separate and maintainable.
"""

# Re-export the main brainio functionality for backwards compatibility
from .brainio import (
    DataAssembly, NeuroidAssembly, BehavioralAssembly, StimulusSet,
    walk_coords, merge_data_arrays, gather_indexes,
    package_data_assembly, package_stimulus_set, write_netcdf,
    load_file_from_s3, load_weight_file, sha1_hash,
    subset, index_efficient
)

__version__ = "1.0.0"
