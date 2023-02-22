#!/bin/bash

LIBRARY_PATH=$1
LIBRARY_NAME=$2
MODEL_ID=$3
BENCHMARK_ID=$4
ENV_NAME=$5
ENVS_DIR=$6

cd "$LIBRARY_PATH" || exit 2
echo "In directory: $PWD"
echo "Setting up conda environment: ${ENV_NAME}"
if [ -d $ENVS_DIR/$ENV_NAME ]; then
	conda env remove -n $ENV_NAME
fi
eval "$(command conda 'shell.bash' 'hook' 2>/dev/null)"
output=$(conda create -n $ENV_NAME python=3.8 -y 2>&1) || echo $output
conda activate $ENV_NAME
output=$(python -m pip install "." 2>&1) || echo $output

echo "Scoring ${MODEL_ID} on ${BENCHMARK_ID}"
python $LIBRARY_NAME score --model_identifier=$MODEL_ID --benchmark_identifier=$BENCHMARK_ID

exit $?
