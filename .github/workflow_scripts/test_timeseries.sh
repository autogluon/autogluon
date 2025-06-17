#!/bin/bash

set -ex

while getopts ":-:" opt; do
  [ "$opt" = "-" ] && [ "${OPTARG}" = "is-platform-test" ] && IS_PLATFORM_TEST=1
done

source $(dirname "$0")/env_setup.sh

setup_build_env
export CUDA_VISIBLE_DEVICES=0
install_local_packages "common/[tests]" "core/[all,tests]" "features/" "tabular/[all,tests]" "timeseries/[all,tests]"
python -m pip install --upgrade pytest-xdist

export PYTHONHASHSEED=0  # for consistency in xdist tests

# Use wheel bundled CUDA instead of DLC CUDA with fallback to compatibility check bypass
PYTORCH_CUDA_PATH=$(python -c "import torch, sys; torch_cuda_path=''; try: torch_cuda_path=torch._C._cuda_getLibPath(); print(torch_cuda_path if torch_cuda_path else ''); except: print('')")

if [ -n "$PYTORCH_CUDA_PATH" ]; then
    echo "Using PyTorch bundled CUDA libraries from: $PYTORCH_CUDA_PATH"
    export LD_LIBRARY_PATH=$PYTORCH_CUDA_PATH:$LD_LIBRARY_PATH
else
    echo "Warning: Could not get PyTorch bundled CUDA path. Falling back to PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK=1"
    export PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK=1
fi

cd timeseries/
if [ "$IS_PLATFORM_TEST" = 1 ]; then
    python -m pytest --junitxml=results.xml --runslow tests  # run platform tests without multiprocessing
else
    python -m pytest --junitxml=results.xml --runslow --numprocesses 4 tests
fi
