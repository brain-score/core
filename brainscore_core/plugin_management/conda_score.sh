#!/bin/bash

LIBRARY_ROOT=$1
MODEL_ID=$2
BENCHMARK_ID=$3
ENV_NAME=$4

echo "Setting up conda environment: ${ENV_NAME}"
eval "$(command conda 'shell.bash' 'hook' 2>/dev/null)"
output=$(conda create -n $ENV_NAME python=3.8 -y 2>&1) || echo $output
conda activate $ENV_NAME
output=$(python -m pip install "." 2>&1) || echo $output

echo "Scoring ${MODEL_ID} on ${BENCHMARK_ID}"
python $LIBRARY_ROOT score --model_identifier=$MODEL_ID --benchmark_identifier=$BENCHMARK_ID

exit $?
