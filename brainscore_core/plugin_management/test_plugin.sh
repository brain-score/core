#!/bin/bash

PLUGIN_PATH=$1
PLUGIN_NAME=$2
PLUGIN_REQUIREMENTS_PATH=$PLUGIN_PATH/requirements.txt
PLUGIN_TEST_PATH=$PLUGIN_PATH/test.py
SINGLE_TEST=$3
CONDA_ENV_PATH=$PLUGIN_PATH/environment.yml
LIBRARY_PATH=$4

PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")

TRAVIS_PYTEST_SETTINGS=${PYTEST_SETTINGS:-"not requires_gpu and not memory_intense and not slow and not travis_slow"}
PYTEST_SETTINGS=${PYTEST_SETTINGS:-"not slow"}

cd "$LIBRARY_PATH" || exit 2
echo "$PLUGIN_NAME ($PLUGIN_PATH)"

### DEPENDENCIES
echo "Setting up conda environment..."
eval "$(command conda 'shell.bash' 'hook' 2>/dev/null)"
output=$(conda create -n $PLUGIN_NAME python=$PYTHON_VERSION -y 2>&1)
conda activate $PLUGIN_NAME
if [ -f "$CONDA_ENV_PATH" ]; then
  output=$(conda env update --file $CONDA_ENV_PATH 2>&1)
fi
if [ -f "$PLUGIN_REQUIREMENTS_PATH" ]; then
  output=$(pip install -r $PLUGIN_REQUIREMENTS_PATH 2>&1)
fi

output=$(python -m pip install -e ".[test]" 2>&1) # install library requirements

### RUN TESTING
if [ "$SINGLE_TEST" != False ]; then
  echo "Running ${SINGLE_TEST}"
  pytest -m "$PYTEST_SETTINGS" "-vv" $PLUGIN_TEST_PATH "-k" $SINGLE_TEST "--log-cli-level=INFO"
else
  if [ "${TRAVIS}" ]; then
    if [ "$PRIVATE_ACCESS" = 1 ]; then
      pytest -m "private_access and $TRAVIS_PYTEST_SETTINGS" $PLUGIN_TEST_PATH; 
    fi
    if [ "$PRIVATE_ACCESS" != 1 ]; then 
      pytest -m "not private_access and $TRAVIS_PYTEST_SETTINGS" $PLUGIN_TEST_PATH; 
    fi
  fi
  elif [ "${OPENMIND}" ]; then
    PLUGIN_XML_FILE="$PLUGIN_NAME"_"$XML_FILE"
    echo "<?xml version=\"1.0\" encoding=\"UTF-8\"?><testsuites></testsuites>" > $PLUGIN_XML_FILE
    pytest -m "$PYTEST_SETTINGS" $PLUGIN_TEST_PATH --junitxml $PLUGIN_XML_FILE --capture=no -o log_cli=true;
    junitparser merge $XML_FILE $PLUGIN_XML_FILE $XML_FILE
    rm $PLUGIN_XML_FILE
  else
    pytest -m "$PYTEST_SETTINGS" $PLUGIN_TEST_PATH;
  fi 
fi

exit $?
