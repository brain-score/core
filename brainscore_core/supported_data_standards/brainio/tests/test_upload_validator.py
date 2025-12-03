#!/usr/bin/env python3
"""
Test script for the upload validator.
Tests validation of stimulus sets and assemblies with various scenarios.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from brainscore_core.supported_data_standards.brainio.upload_validator import (
    validate_stimulus_set_only,
    validate_assembly_only,
    validate_packaging_data,
    ValidationError
)
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_core.supported_data_standards.brainio.packaging import package_stimulus_set_locally, package_data_assembly_locally


def create_valid_stimulus_set():
    """Create a valid stimulus set for testing."""
    # Create dummy images
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    for i in range(3):
        img_path = os.path.join(temp_dir, f'image_{i}.png')
        # Create a simple dummy image file
        with open(img_path, 'w') as f:
            f.write('dummy image data')
        image_paths.append(img_path)
    
    # Create stimulus set
    stimulus_data = {
        'stimulus_id': [f'image_{i}' for i in range(3)],
        'category': ['animal', 'object', 'scene'],
        'complexity': [1.2, 2.5, 3.1]
    }
    
    stimulus_set = StimulusSet(stimulus_data)
    stimulus_set.stimulus_paths = {f'image_{i}': image_paths[i] for i in range(3)}
    
    def get_stimulus(stimulus_id):
        return stimulus_set.stimulus_paths[stimulus_id]
    
    stimulus_set.get_stimulus = get_stimulus
    return stimulus_set, temp_dir


def create_valid_assembly():
    """Create a valid assembly for testing."""
    n_presentations = 6
    n_neurons = 4
    
    # Create coordinates
    presentation_ids = [f'presentation_{i}' for i in range(n_presentations)]
    stimulus_ids = [f'image_{i % 3}' for i in range(n_presentations)]  # cycle through 3 images
    neuroid_ids = [f'neuron_{i}' for i in range(n_neurons)]
    
    # Create dummy neural data
    neural_data = np.random.randn(n_presentations, n_neurons)
    
    # Create the DataArray
    assembly = xr.DataArray(
        neural_data,
        coords={
            'presentation': presentation_ids,
            'neuroid': neuroid_ids,
            'stimulus_id': ('presentation', stimulus_ids),
            'repetition': ('presentation', [i % 2 for i in range(n_presentations)]),
            'region': ('neuroid', ['V1', 'V2', 'V4', 'IT'])
        },
        dims=['presentation', 'neuroid']
    )
    
    return assembly


def create_invalid_stimulus_set_no_stimulus_id():
    """Create an invalid stimulus set missing stimulus_id column."""
    stimulus_data = {
        'image_id': [f'image_{i}' for i in range(3)],
        'category': ['animal', 'object', 'scene']
    }
    
    stimulus_set = StimulusSet(stimulus_data)
    # Add required attributes but with wrong column name
    stimulus_set.stimulus_paths = {f'image_{i}': f'/tmp/image_{i}.png' for i in range(3)}
    return stimulus_set


def create_invalid_stimulus_set_no_other_columns():
    """Create an invalid stimulus set with only stimulus_id column."""
    stimulus_data = {
        'stimulus_id': [f'image_{i}' for i in range(3)]
    }
    
    stimulus_set = StimulusSet(stimulus_data)
    # Add required attributes
    stimulus_set.stimulus_paths = {f'image_{i}': f'/tmp/image_{i}.png' for i in range(3)}
    return stimulus_set


def create_invalid_assembly_no_stimulus_id():
    """Create an invalid assembly missing stimulus_id coordinate."""
    n_presentations = 6
    n_neurons = 4
    
    presentation_ids = [f'presentation_{i}' for i in range(n_presentations)]
    neuroid_ids = [f'neuron_{i}' for i in range(n_neurons)]
    
    neural_data = np.random.randn(n_presentations, n_neurons)
    
    assembly = xr.DataArray(
        neural_data,
        coords={
            'presentation': presentation_ids,
            'neuroid': neuroid_ids,
            'repetition': ('presentation', [i % 2 for i in range(n_presentations)])
        },
        dims=['presentation', 'neuroid']
    )
    
    return assembly


def test_valid_stimulus_set():
    """Test validation of a valid stimulus set."""
    print("Testing valid stimulus set...")
    
    stimulus_set, temp_dir = create_valid_stimulus_set()
    
    try:
        result = validate_stimulus_set_only(stimulus_set, "valid_stimulus_set")
        print(f"Valid stimulus set passed validation: {result}")
        return True
    except Exception as e:
        print(f"Valid stimulus set failed validation: {e}")
        return False
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_invalid_stimulus_set_no_stimulus_id():
    """Test validation of invalid stimulus set missing stimulus_id."""
    print("Testing invalid stimulus set (no stimulus_id)...")
    
    stimulus_set = create_invalid_stimulus_set_no_stimulus_id()
    
    try:
        validate_stimulus_set_only(stimulus_set, "invalid_stimulus_set")
        print("Invalid stimulus set should have failed validation")
        return False
    except ValidationError as e:
        print(f"Invalid stimulus set correctly failed validation: {e}")
        return True
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def test_invalid_stimulus_set_no_other_columns():
    """Test validation of invalid stimulus set with only stimulus_id."""
    print("Testing invalid stimulus set (no other columns)...")
    
    stimulus_set = create_invalid_stimulus_set_no_other_columns()
    
    try:
        validate_stimulus_set_only(stimulus_set, "invalid_stimulus_set")
        print("Invalid stimulus set should have failed validation")
        return False
    except ValidationError as e:
        print(f"Invalid stimulus set correctly failed validation: {e}")
        return True
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def test_valid_assembly():
    """Test validation of a valid assembly."""
    print("Testing valid assembly...")
    
    assembly = create_valid_assembly()
    
    try:
        result = validate_assembly_only(assembly, "valid_assembly")
        print(f"Valid assembly passed validation: {result}")
        return True
    except Exception as e:
        print(f"Valid assembly failed validation: {e}")
        return False


def test_invalid_assembly_no_stimulus_id():
    """Test validation of invalid assembly missing stimulus_id."""
    print("Testing invalid assembly (no stimulus_id)...")
    
    assembly = create_invalid_assembly_no_stimulus_id()
    
    try:
        validate_assembly_only(assembly, "invalid_assembly")
        print("Invalid assembly should have failed validation")
        return False
    except ValidationError as e:
        print(f"Invalid assembly correctly failed validation: {e}")
        return True
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def test_packaging_with_validation():
    """Test that packaging functions use validation."""
    print("Testing packaging with validation...")
    
    # Test with valid data
    stimulus_set, temp_dir = create_valid_stimulus_set()
    assembly = create_valid_assembly()
    
    try:
        with tempfile.TemporaryDirectory() as temp_output:
            # Test stimulus set packaging
            result1 = package_stimulus_set_locally(
                proto_stimulus_set=stimulus_set,
                stimulus_set_identifier="test.valid.2024",
                downloads_path=temp_output
            )
            print(f"Valid stimulus set packaged successfully")
            
            # Test assembly packaging
            result2 = package_data_assembly_locally(
                proto_data_assembly=assembly,
                assembly_identifier="test.valid.2024",
                stimulus_set_identifier="test.valid.2024",
                assembly_class_name="NeuroidAssembly",  # Use available class
                downloads_path=temp_output
            )
            print(f"Valid assembly packaged successfully")
            
        return True
    except Exception as e:
        print(f"Valid data packaging failed: {e}")
        return False
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_packaging_with_invalid_data():
    """Test that packaging functions reject invalid data."""
    print("Testing packaging with invalid data...")
    
    # Test with invalid stimulus set
    invalid_stimulus_set = create_invalid_stimulus_set_no_stimulus_id()
    
    try:
        with tempfile.TemporaryDirectory() as temp_output:
            package_stimulus_set_locally(
                proto_stimulus_set=invalid_stimulus_set,
                stimulus_set_identifier="test.invalid.2024",
                downloads_path=temp_output
            )
            print("Invalid stimulus set should have been rejected")
            return False
    except ValueError as e:
        print(f"Invalid stimulus set correctly rejected: {e}")
        return True
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def main():
    """Run all validation tests."""
    print("Testing Upload Validator")
    print("=" * 50)
    
    tests = [
        test_valid_stimulus_set,
        test_invalid_stimulus_set_no_stimulus_id,
        test_invalid_stimulus_set_no_other_columns,
        test_valid_assembly,
        test_invalid_assembly_no_stimulus_id,
        test_packaging_with_validation,
        test_packaging_with_invalid_data
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()
        if test():
            passed += 1
        print("-" * 30)
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! Validator is working correctly.")
        return 0
    else:
        print("Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    exit(main())
