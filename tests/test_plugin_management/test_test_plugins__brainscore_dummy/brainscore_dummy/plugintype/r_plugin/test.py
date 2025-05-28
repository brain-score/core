import os
import sys
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects


def test_r():
    """Test that R can be initialized and basic R functionality works."""
    try:
        # Initialize R
        robjects.r('Sys.setenv(LANG = "en_US.UTF-8")')  # Set locale to avoid encoding issues
        
        # Try to import base package
        base = importr('base')
        
        # Test basic R functionality
        r_version = base.R_version()
        print(f"R version: {r_version[0]}")
        
        # Test a simple R operation
        result = robjects.r('1 + 1')
        assert float(result[0]) == 2.0, "Basic R arithmetic failed"
        
    except Exception as e:
        print(f"Error initializing R: {e}", file=sys.stderr)
        print(f"R_HOME: {os.environ.get('R_HOME', 'not set')}", file=sys.stderr)
        print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'not set')}", file=sys.stderr)
        raise
