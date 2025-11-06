#!/bin/bash

set -ex

while getopts ":-:" opt; do
  [ "$opt" = "-" ] && [ "${OPTARG}" = "is-platform-test" ] && IS_PLATFORM_TEST=1
done

source $(dirname "$0")/env_setup.sh

setup_build_env
export CUDA_VISIBLE_DEVICES=0
install_local_packages "common/[tests]" "core/[all,tests]" "features/" "tabular/[all,tests]" "timeseries/[all,tests]"
setup_pytorch_cuda_env
python -m pip install --upgrade pytest-xdist

export PYTHONHASHSEED=0  # for consistency in xdist tests
export PYTORCH_ENABLE_MPS_FALLBACK=1  # for MacOS compatibility
export OMP_NUM_THREADS=1  # prevent thread oversubscription in parallel tests

cd timeseries/
if [ "$IS_PLATFORM_TEST" = 1 ]; then
    python -m pytest --junitxml=results.xml --runslow tests  # run platform tests without multiprocessing
else
    # Set PyTorch multiprocessing start method to 'spawn' to avoid CUDA fork issues
    python -m pytest --junitxml=results.xml --runslow --numprocesses 4 -o python_functions="test_*" tests
fi
