"""
BRAINIO DATA STANDARD - Brain-Score's native data format

This module provides Brain-Score's native data format support, extracted from
the original BrainIO library. It handles assemblies (neural/behavioral data)
and stimulus sets with S3 storage integration.

PURPOSE:
--------
Brain-Score was using the full BrainIO library but only needed about 80% of its functionality.
This module extracts the essential parts and removes the catalog system, lookup tables, 
and unused validation. The original BrainIO library was unmaintained, so Brain-Score
absorbed the necessary functionality to remain self-contained.

WHAT'S INCLUDED:
---------------
- Data structures: DataAssembly, NeuroidAssembly, BehavioralAssembly, StimulusSet
- Upload/download: S3 operations for assemblies and stimulus sets  
- File operations: NetCDF writing, ZIP handling, SHA1 verification
- Data utilities: Coordinate walking, array merging, transformations

WHAT'S REMOVED:
--------------
- Catalog system (CSV-based lookup tables)
- Unused assembly types and complex validation
- Legacy compatibility code not used by Brain-Score
"""

from .fetch import get_assembly, get_stimulus_set
from .assemblies import (
    DataAssembly, NeuroidAssembly, BehavioralAssembly, 
    walk_coords, merge_data_arrays, gather_indexes
)
from .stimuli import StimulusSet
from .packaging import package_data_assembly, package_stimulus_set, write_netcdf
from .s3 import load_file_from_s3, load_weight_file, sha1_hash
from .transform import subset, index_efficient

__version__ = "1.0.0"
