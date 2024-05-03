#!/bin/bash

PLUGIN_PATH=$1
PLUGIN_NAME=$2
PLUGIN_SETUP_PATH=$PLUGIN_PATH/setup.py
PLUGIN_REQUIREMENTS_PATH=$PLUGIN_PATH/requirements.txt
PLUGIN_TEST_PATH=$PLUGIN_PATH/test.py
SINGLE_TEST=$3
CONDA_ENV_PATH=$PLUGIN_PATH/environment.yml
LIBRARY_PATH=$4
GENERIC_TEST_PATH=$5

PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")

TRAVIS_PYTEST_SETTINGS=${PYTEST_SETTINGS:-"not requires_gpu and not memory_intense and not slow and not travis_slow"}
PYTEST_SETTINGS=${PYTEST_SETTINGS:-"not slow"}

cd "$LIBRARY_PATH" || exit 2
echo "$PLUGIN_NAME ($PLUGIN_PATH)"

### DEPENDENCIES
echo "Setting up conda environment..."
echo "Python version: $PYTHON_VERSION"
dir_path=$(pwd)
echo "Current directory: $dir_path"
eval "$(command conda 'shell.bash' 'hook' 2>/dev/null)"

# reconstruct previously created conda env name
prev_env_name=$(basename $(dirname "$dir_path"))
echo "Detected Conda environment name: $prev_env_name"
# Use awk to split the directory name and rearrange it to match the required environment name format
env_name=$(echo "$prev_env_name" | awk -F'_' '{print $3 "_unittest_plugins_" $4}')
echo "Constructed Conda environment name: $env_name"

# clone current env
conda create -n $env_name --clone $prev_env_name -y 2>&1
if [ $? -ne 0 ]; then
  echo "Failed to create environment: $env_name"
  exit 1
fi
conda activate $PLUGIN_NAME

if [ -f "$CONDA_ENV_PATH" ]; then
  output=$(conda env update --file $CONDA_ENV_PATH 2>&1)
fi
if [ -f "$PLUGIN_SETUP_PATH" ]; then
  output=$(pip install $PLUGIN_PATH 2>&1)
fi
if [ -f "$PLUGIN_REQUIREMENTS_PATH" ]; then
  output=$(pip install -r $PLUGIN_REQUIREMENTS_PATH 2>&1)
fi

### RUN GENERIC TESTING
if [ "$GENERIC_TEST_PATH" != False ]; then
  pytest -m "$PYTEST_SETTINGS" "-vv" $GENERIC_TEST_PATH "--plugin_directory" $PLUGIN_PATH "--log-cli-level=INFO" "--junitxml" $XML_FILE
fi

### RUN TESTING
if [ "$SINGLE_TEST" != False ]; then
  echo "Running ${SINGLE_TEST}"
  pytest -m "$PYTEST_SETTINGS" "-vv" $PLUGIN_TEST_PATH "-k" $SINGLE_TEST "--log-cli-level=INFO"
else
  if [ "${TRAVIS}" ]; then
    if [ "$PRIVATE_ACCESS" = 1 ]; then
      pytest -m "private_access and $TRAVIS_PYTEST_SETTINGS" $PLUGIN_TEST_PATH; 
    elif [ "$PRIVATE_ACCESS" != 1 ]; then 
      pytest -m "not private_access and $TRAVIS_PYTEST_SETTINGS" $PLUGIN_TEST_PATH; 
    fi
  elif [ "${OPENMIND}" ]; then
    pip install junitparser
    PLUGIN_XML_FILE="$PLUGIN_NAME"_"$XML_FILE"
    pytest -m "$PYTEST_SETTINGS" $PLUGIN_TEST_PATH --junitxml $PLUGIN_XML_FILE --capture=no -o log_cli=true;
    junitparser merge $XML_FILE $PLUGIN_XML_FILE $XML_FILE
    rm $PLUGIN_XML_FILE
  else
    pytest -m "$PYTEST_SETTINGS" $PLUGIN_TEST_PATH;
  fi 
fi

exit $?
