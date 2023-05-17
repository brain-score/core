#!/bin/bash

LIBRARY_PATH=$1
LIBRARY_NAME=$2
MODEL_ID=$3
BENCHMARK_ID=$4
ENV_NAME=$5
ENVS_DIR=$6

### DEPENDENCIES

get_plugin_dir() {
  python brainscore_core/plugin_management/import_plugin print_plugin_dir "$LIBRARY_NAME" "$1" "$2"
}

MODEL_DIR=$LIBRARY_NAME/models/$(get_plugin_dir "models" "$MODEL_ID")
BENCHMARK_DIR=$LIBRARY_NAME/benchmarks/$(get_plugin_dir "benchmarks" "$BENCHMARK_ID")

MODEL_ENV_YML=$MODEL_DIR/environment.yml
BENCHMARK_ENV_YML=$BENCHMARK_DIR/environment.yml

cd "$LIBRARY_PATH" || exit 2
echo "In directory: $PWD"
if [ -d $ENVS_DIR/$ENV_NAME ]; then
  conda env remove -n $ENV_NAME
fi
echo "Setting up conda environment: ${ENV_NAME}"
eval "$(command conda 'shell.bash' 'hook' 2>/dev/null)"
output=$(conda create -n $ENV_NAME python=3.8 -y 2>&1) || echo $output
conda activate $ENV_NAME
# install plugin yml environments if available
if [ -f "$MODEL_ENV_YML" ]; then
  output=$(conda env update --file $MODEL_ENV_YML 2>&1) || echo $output
fi
if [ -f "$BENCHMARK_ENV_YML" ]; then
  output=$(conda env update --file $BENCHMARK_ENV_YML 2>&1) || echo $output
fi
# install library dependencies
output=$(python -m pip install "." 2>&1) || echo $output

### SCORING
echo "Scoring ${MODEL_ID} on ${BENCHMARK_ID}"
python $LIBRARY_NAME score --model_identifier=$MODEL_ID --benchmark_identifier=$BENCHMARK_ID --conda_active=True

exit $?
