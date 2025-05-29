import os
import sys
import traceback
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.rinterface as rinterface


def test_r():
    """Test that R can be initialized and basic R functionality works."""
    print("Starting R test...", file=sys.stderr)
    print(f"Python version: {sys.version}", file=sys.stderr)
    print(f"Current working directory: {os.getcwd()}", file=sys.stderr)
    print(f"Environment variables:", file=sys.stderr)
    for var in ['R_HOME', 'LD_LIBRARY_PATH', 'PATH', 'CONDA_PREFIX']:
        print(f"  {var}: {os.environ.get(var, 'not set')}", file=sys.stderr)
    
    try:
        print("Attempting to initialize R...", file=sys.stderr)
        # Try to get R version before any other operations
        try:
            r_version = rinterface.get_rversion()
            print(f"R version from rinterface: {r_version}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to get R version from rinterface: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        
        print("Setting R locale...", file=sys.stderr)
        robjects.r('Sys.setenv(LANG = "en_US.UTF-8")')
        
        print("Importing base package...", file=sys.stderr)
        base = importr('base')
        
        print("Getting R version from base package...", file=sys.stderr)
        r_version = base.R_version()
        print(f"R version from base package: {r_version[0]}", file=sys.stderr)
        
        print("Testing basic R operation...", file=sys.stderr)
        result = robjects.r('1 + 1')
        print(f"R operation result: {result[0]}", file=sys.stderr)
        assert float(result[0]) == 2.0, "Basic R arithmetic failed"
        
        print("R test completed successfully", file=sys.stderr)
        
    except Exception as e:
        print(f"Error in R test: {e}", file=sys.stderr)
        print("Full traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        
        # Try to get more system information
        try:
            import subprocess
            print("\nSystem information:", file=sys.stderr)
            subprocess.run(['ldd', sys.executable], capture_output=True, text=True, check=True)
            print("\nR library information:", file=sys.stderr)
            subprocess.run(['ldd', os.path.join(os.environ.get('CONDA_PREFIX', ''), 'lib/R/lib/libR.so')], 
                         capture_output=True, text=True, check=True)
        except Exception as sys_e:
            print(f"Failed to get system information: {sys_e}", file=sys.stderr)
        
        raise
