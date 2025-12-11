"""
Upload Validator - Validates stimulus sets and assemblies before packaging.

This module provides validation functions to ensure that stimulus sets and
assemblies are properly formatted and compatible before packaging locally
or uploading to S3.

VALIDATION RULES:
----------------
1. Stimulus Set must have 'stimulus_id' column and at least one other column
2. Assembly must have 'stimulus_id' column and at least one other column  
3. If columns exist in both SS and assembly, they must have the same data type
4. All required metadata must be present and valid

PURPOSE:
--------
Prevents packaging of invalid or incompatible data that could cause issues
downstream in the Brain-Score ecosystem.
"""

import logging
from typing import Dict, List, Set, Tuple, Any
import pandas as pd
import xarray as xr
from brainscore_core.supported_data_standards.brainio.assemblies import get_metadata

_logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_stimulus_set(stimulus_set, identifier="stimulus_set") -> bool:
    """
    Validate a stimulus set meets basic requirements.
    
    Args:
        stimulus_set: StimulusSet object to validate
        identifier: Name for error messages (default: "stimulus_set")
    
    Returns:
        bool: True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    _logger.debug(f"Validating {identifier}...")
    
    # Check if stimulus_set has required attributes
    if not hasattr(stimulus_set, 'columns'):
        raise ValidationError(f"{identifier}: Missing 'columns' attribute")
    
    if not hasattr(stimulus_set, 'stimulus_paths'):
        raise ValidationError(f"{identifier}: Missing 'stimulus_paths' attribute")
    
    # Check for stimulus_id column
    if 'stimulus_id' not in stimulus_set.columns:
        raise ValidationError(f"{identifier}: Missing required 'stimulus_id' column")
    
    # Check for at least one other column
    other_columns = [col for col in stimulus_set.columns if col != 'stimulus_id']
    if len(other_columns) == 0:
        raise ValidationError(f"{identifier}: Must have at least one column besides 'stimulus_id'")
    
    # Check that stimulus_set is not empty
    if len(stimulus_set) == 0:
        raise ValidationError(f"{identifier}: Cannot be empty")
    
    # Check that all stimulus_ids have corresponding paths
    missing_paths = []
    for stimulus_id in stimulus_set['stimulus_id']:
        if stimulus_id not in stimulus_set.stimulus_paths:
            missing_paths.append(stimulus_id)
    
    if missing_paths:
        raise ValidationError(f"{identifier}: Missing stimulus paths for IDs: {missing_paths[:5]}{'...' if len(missing_paths) > 5 else ''}")
    
    _logger.debug(f"{identifier}: Validation passed")
    return True


def validate_assembly(assembly, identifier="assembly") -> bool:
    """
    Validate an assembly meets basic requirements.
    
    Args:
        assembly: DataAssembly object to validate
        identifier: Name for error messages (default: "assembly")
    
    Returns:
        bool: True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    _logger.debug(f"Validating {identifier}...")
    
    # Check if assembly has required attributes
    if not hasattr(assembly, 'coords'):
        raise ValidationError(f"{identifier}: Missing 'coords' attribute")
    
    # Check for stimulus_id coordinate
    if not hasattr(assembly, 'stimulus_id') :
        raise ValidationError(f"{identifier}: Missing required 'stimulus_id' coordinate")
    
    # Check for at least one other coordinate (besides stimulus_id)
    other_coords = [coord for coord in assembly.coords if coord != 'stimulus_id']
    if len(other_coords) == 0:
        raise ValidationError(f"{identifier}: Must have at least one coordinate besides 'stimulus_id'")
    
    # Check that assembly is not empty
    if assembly.size == 0:
        raise ValidationError(f"{identifier}: Cannot be empty")
    
    # Check that stimulus_id coordinate is not empty
    stimulus_ids = assembly.coords['stimulus_id']
    if len(stimulus_ids) == 0:
        raise ValidationError(f"{identifier}: 'stimulus_id' coordinate cannot be empty")
    
    _logger.debug(f"{identifier}: Validation passed")
    return True


def get_common_columns(stimulus_set, assembly) -> Set[str]:
    """
    Find columns that exist in both stimulus set and assembly.
    
    Args:
        stimulus_set: StimulusSet object
        assembly: DataAssembly object
    
    Returns:
        Set of common column names
    """
    stimulus_columns = set(stimulus_set.columns)
    assembly_coords = set(get_metadata(assembly, names_only=True, include_coords=True, include_levels=True))
    
    # Find intersection (common columns)
    common_columns = stimulus_columns.intersection(assembly_coords)
    
    _logger.debug(f"Common columns between stimulus set and assembly: {common_columns}")
    return common_columns


def validate_data_type_compatibility(stimulus_set, assembly, common_columns: Set[str]) -> bool:
    """
    Validate that common columns have compatible data types.
    
    Args:
        stimulus_set: StimulusSet object
        assembly: DataAssembly object
        common_columns: Set of column names that exist in both
    
    Returns:
        bool: True if all data types are compatible
        
    Raises:
        ValidationError: If data types are incompatible
    """
    _logger.debug(f"Validating data type compatibility for {len(common_columns)} common columns...")
    
    for column in common_columns:
        # Get data types
        stimulus_dtype = stimulus_set[column].dtype
        assembly_dtype = assembly.coords[column].dtype
        
        # Check if data types are compatible
        if not _are_dtypes_compatible(stimulus_dtype, assembly_dtype):
            raise ValidationError(
                f"Data type mismatch for column '{column}': "
                f"stimulus_set has {stimulus_dtype}, assembly has {assembly_dtype}"
            )
    
    _logger.debug("Data type compatibility validation passed")
    return True


def _are_dtypes_compatible(dtype1, dtype2) -> bool:
    """
    Check if two pandas/xarray dtypes are compatible.
    
    Args:
        dtype1: First dtype
        dtype2: Second dtype
    
    Returns:
        bool: True if dtypes are compatible
    """
    # Convert to string for comparison
    str1 = str(dtype1)
    str2 = str(dtype2)
    
    # If they're exactly the same, they're compatible
    if str1 == str2:
        return True
    
    # Check for common compatible types
    compatible_groups = [
        ['int64', 'int32', 'int16', 'int8'],
        ['float64', 'float32', 'float16'],
        ['object', 'string'],
        ['bool', 'boolean']
    ]
    
    for group in compatible_groups:
        if str1 in group and str2 in group:
            return True
    
    # For categorical data, check if they're both categorical
    if 'categor' in str1.lower() and 'categor' in str2.lower():
        return True
    
    return False


def validate_stimulus_set_assembly_compatibility(stimulus_set, assembly) -> bool:
    """
    Validate that stimulus set and assembly are compatible.
    
    Args:
        stimulus_set: StimulusSet object
        assembly: DataAssembly object
    
    Returns:
        bool: True if compatible
        
    Raises:
        ValidationError: If incompatible
    """
    _logger.debug("Validating stimulus set and assembly compatibility...")
    
    # Find common columns
    common_columns = get_common_columns(stimulus_set, assembly)
    
    # Validate data type compatibility for common columns
    if common_columns:
        validate_data_type_compatibility(stimulus_set, assembly, common_columns)
    
    # Check that stimulus_ids in assembly exist in stimulus_set
    assembly_stimulus_ids = set(assembly.coords['stimulus_id'].values)
    stimulus_set_ids = set(stimulus_set['stimulus_id'].values)
    
    missing_in_stimulus_set = assembly_stimulus_ids - stimulus_set_ids
    if missing_in_stimulus_ids:
        raise ValidationError(
            f"Assembly references stimulus_ids not in stimulus set: {list(missing_in_stimulus_ids)[:5]}{'...' if len(missing_in_stimulus_ids) > 5 else ''}"
        )
    
    _logger.debug("Stimulus set and assembly compatibility validation passed")
    return True


def validate_packaging_data(stimulus_set, assembly, stimulus_set_identifier="stimulus_set", assembly_identifier="assembly") -> bool:
    """
    Comprehensive validation for packaging stimulus set and assembly together.
    
    Args:
        stimulus_set: StimulusSet object
        assembly: DataAssembly object  
        stimulus_set_identifier: Name for stimulus set in error messages
        assembly_identifier: Name for assembly in error messages
    
    Returns:
        bool: True if all validations pass
        
    Raises:
        ValidationError: If any validation fails
    """
    _logger.info(f"Starting comprehensive validation for {stimulus_set_identifier} and {assembly_identifier}")
    
    try:
        # Validate individual components
        validate_stimulus_set(stimulus_set, stimulus_set_identifier)
        validate_assembly(assembly, assembly_identifier)
        
        # Validate compatibility
        validate_stimulus_set_assembly_compatibility(stimulus_set, assembly)
        
        _logger.info("All validations passed successfully")
        return True
        
    except ValidationError as e:
        _logger.error(f"Validation failed: {e}")
        raise
    except Exception as e:
        _logger.error(f"Unexpected error during validation: {e}")
        raise ValidationError(f"Unexpected validation error: {e}")


def validate_stimulus_set_only(stimulus_set, identifier="stimulus_set") -> bool:
    """
    Validate only a stimulus set (for cases where no assembly is provided).
    
    Args:
        stimulus_set: StimulusSet object
        identifier: Name for error messages
    
    Returns:
        bool: True if valid
    """
    return validate_stimulus_set(stimulus_set, identifier)


def validate_assembly_only(assembly, identifier="assembly") -> bool:
    """
    Validate only an assembly (for cases where no stimulus set is provided).
    
    Args:
        assembly: DataAssembly object
        identifier: Name for error messages
    
    Returns:
        bool: True if valid
    """
    return validate_assembly(assembly, identifier)
