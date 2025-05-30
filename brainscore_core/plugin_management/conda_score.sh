#!/bin/bash

# Script to run model scoring in an isolated conda environment
# Usage: conda_score.sh <library_path> <library_name> <model_id> <benchmark_id> <env_name> <envs_dir>

# Input parameters
LIBRARY_PATH="$1"      # Path to the library directory
LIBRARY_NAME="$2"      # Name of the library (e.g., brainscore_dummy)
MODEL_ID="$3"         # Model identifier
BENCHMARK_ID="$4"     # Benchmark identifier
ENV_NAME="$5"         # Name for the conda environment
ENVS_DIR="$6"         # Directory for conda environments

# Get current Python version for the new environment
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "Using Python version: $PYTHON_VERSION"

# Function to get plugin directory using the brainscore plugin management system
get_plugin_dir() {
    local plugin_type="$1"
    local plugin_id="$2"
    python -m brainscore_core.plugin_management.import_plugin print_plugin_dir "$LIBRARY_NAME" "$plugin_type" "$plugin_id" || {
        echo "Warning: Could not retrieve plugin directory for $plugin_type/$plugin_id"
        return 1
    }
}

# Determine plugin-specific directories and environment files
MODEL_PLUGIN_DIR=$(get_plugin_dir "models" "$MODEL_ID")
BENCHMARK_PLUGIN_DIR=$(get_plugin_dir "benchmarks" "$BENCHMARK_ID")

# Remove existing environment if it exists
if [ -d "$ENVS_DIR/$ENV_NAME" ]; then
    echo "Removing existing environment: $ENV_NAME"
    conda env remove -n "$ENV_NAME" -y
fi

# Create new conda environment
echo "Creating new conda environment: $ENV_NAME"
eval "$(conda shell.bash hook)"
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y || {
    echo "Error: Failed to create conda environment"
    exit 1
}

# Activate environment and install pip
conda activate "$ENV_NAME" || {
    echo "Error: Failed to activate conda environment"
    exit 1
}
conda install pip -y || {
    echo "Error: Failed to install pip"
    exit 1
}

# Change to the library directory
cd "$LIBRARY_PATH/$LIBRARY_NAME" || {
    echo "Error: Could not change to directory $LIBRARY_PATH/$LIBRARY_NAME"
    exit 1
}

# Install the library in editable mode
echo "Installing library in editable mode"
if [ -f "requirements.txt" ]; then
    echo "Found requirements.txt, installing dependencies..."
    pip install -r requirements.txt || {
        echo "Error: Failed to install from requirements.txt"
        exit 1
    }
else
    pip install -e . || {
        echo "Error: Failed to install library"
        exit 1
    }
fi

# Install plugin specific dependencies
if [ -n "$MODEL_PLUGIN_DIR" ]; then
    MODEL_DIR="$LIBRARY_NAME/models/$MODEL_PLUGIN_DIR"
    MODEL_ENV_YML="$MODEL_DIR/environment.yml"
    echo "Model directory: $MODEL_DIR"
else
    echo "Warning: Could not determine model plugin directory"
    MODEL_ENV_YML=""
fi

if [ -n "$BENCHMARK_PLUGIN_DIR" ]; then
    BENCHMARK_DIR="$LIBRARY_NAME/benchmarks/$BENCHMARK_PLUGIN_DIR"
    BENCHMARK_ENV_YML="$BENCHMARK_DIR/environment.yml"
    echo "Benchmark directory: $BENCHMARK_DIR"
else
    echo "Warning: Could not determine benchmark plugin directory"
    BENCHMARK_ENV_YML=""
fi

# Run the scoring command
echo "Running scoring command"
python -m "$LIBRARY_NAME" score \
    --model_identifier="$MODEL_ID" \
    --benchmark_identifier="$BENCHMARK_ID" \
    --conda_active=True

# Capture and return the exit code
EXIT_CODE=$?
echo "Scoring command exited with code: $EXIT_CODE"
exit $EXIT_CODE
