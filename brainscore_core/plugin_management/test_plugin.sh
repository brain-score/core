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
PLUGIN_XML_FILE="$PLUGIN_NAME"_"$XML_FILE" # XML_FILE comes from Openmind environment

GENERIC_TEST_SUCCESS=0
PLUGIN_TEST_SUCCESS=0

cd "$LIBRARY_PATH" || exit 2
echo "$PLUGIN_NAME ($PLUGIN_PATH)"

### DEPENDENCIES
echo "Setting up conda environment..."
eval "$(command conda 'shell.bash' 'hook' 2>/dev/null)"
output=$(conda create -n $PLUGIN_NAME python=$PYTHON_VERSION -y 2>&1)
conda activate $PLUGIN_NAME
conda install pip
pip install --upgrade pip setuptools

output=$(python -m pip install -e ".[test]" --default-timeout=600 --retries=5 2>&1) # install library requirements

if [ -f "$CONDA_ENV_PATH" ]; then
  conda env update --file $CONDA_ENV_PATH 2>&1
fi
if [ -f "$PLUGIN_SETUP_PATH" ]; then
  pip install $PLUGIN_PATH --default-timeout=600 --retries=5 2>&1
fi
if [ -f "$PLUGIN_REQUIREMENTS_PATH" ]; then
  pip install -r $PLUGIN_REQUIREMENTS_PATH --default-timeout=600 --retries=5 2>&1
fi

output=$(pip install junitparser 2>&1)

### RUN GENERIC TESTING
if [ "$GENERIC_TEST_PATH" != False ]; then
  if [ "${OPENMIND}" ]; then
    pytest -m "$PYTEST_SETTINGS" "-vv" $GENERIC_TEST_PATH "--plugin_directory" $PLUGIN_PATH "--log-cli-level=INFO" "--junitxml" $XML_FILE;
  else
    pytest -m "$PYTEST_SETTINGS" "-vv" $GENERIC_TEST_PATH "--plugin_directory" $PLUGIN_PATH "--log-cli-level=INFO";
  fi
  GENERIC_TEST_SUCCESS=$?
fi

### RUN TESTING
if [ "$SINGLE_TEST" != False ]; then
  echo "Running ${SINGLE_TEST}"
  pytest -m "$PYTEST_SETTINGS" "-vv" $PLUGIN_TEST_PATH "-k" $SINGLE_TEST "--log-cli-level=INFO"
  PLUGIN_TEST_SUCCESS=$?
else
  if [ "${TRAVIS}" ]; then
    if [ "$PRIVATE_ACCESS" = 1 ]; then
      pytest -m "private_access and $TRAVIS_PYTEST_SETTINGS" $PLUGIN_TEST_PATH;
    elif [ "$PRIVATE_ACCESS" != 1 ]; then 
      pytest -m "not private_access and $TRAVIS_PYTEST_SETTINGS" $PLUGIN_TEST_PATH;
    fi
  else
    pytest -m "$PYTEST_SETTINGS" $PLUGIN_TEST_PATH "--junitxml" $PLUGIN_XML_FILE "-s" "-o log_cli=true";
  fi 
  PLUGIN_TEST_SUCCESS=$?
  if [ "${OPENMIND}" ]; then
    junitparser merge $XML_FILE $PLUGIN_XML_FILE $XML_FILE
    rm $PLUGIN_XML_FILE
  fi
fi

(($GENERIC_TEST_SUCCESS == 0)) && echo "Generic tests succeeded" || echo "Generic tests failed, return code $GENERIC_TEST_SUCCESS"
(($PLUGIN_TEST_SUCCESS == 0 || $PLUGIN_TEST_SUCCESS == 5)) && echo "Plugin-specific tests succeeded" || echo "Plugin-specific tests failed, return code $PLUGIN_TEST_SUCCESS"

if [ $GENERIC_TEST_SUCCESS -ne 0 ]; then
  exit "$GENERIC_TEST_SUCCESS"
fi

exit "$PLUGIN_TEST_SUCCESS"
