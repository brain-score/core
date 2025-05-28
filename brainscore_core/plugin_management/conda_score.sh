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

# Print parameters for debugging
echo "Parameters:"
echo "  Library path: $LIBRARY_PATH"
echo "  Library name: $LIBRARY_NAME"
echo "  Model ID: $MODEL_ID"
echo "  Benchmark ID: $BENCHMARK_ID"
echo "  Environment name: $ENV_NAME"
echo "  Environments directory: $ENVS_DIR"

# Get current Python version for the new environment
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Using Python version: $PYTHON_VERSION"

# Change to the library directory
cd "$LIBRARY_PATH/$LIBRARY_NAME" || {
    echo "Error: Could not change to directory $LIBRARY_PATH/$LIBRARY_NAME"
    exit 1
}

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
