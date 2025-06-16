#!/bin/bash

set -ex

ADDITIONAL_TEST_ARGS=$1
IS_PLATFORM_TEST=$2

source $(dirname "$0")/env_setup.sh

setup_build_env

if ! [ "$IS_PLATFORM_TEST" = "true" ]
then
    export CUDA_VISIBLE_DEVICES=0
fi

install_local_packages "common/[tests]" "core/[all,tests]" "features/"

if [ "$IS_PLATFORM_TEST" = "true" ]
then
    install_tabular_platforms "[all,tests]"
    install_multimodal "[tests]"
else
    install_tabular "[all,tests]"
    install_multimodal "[tests]"
fi

# Use wheel bundled CUDA instead of DLC CUDA with fallback to compatibility check bypass
PYTORCH_CUDA_PATH=$(python -c "import torch, sys; torch_cuda_path=''; try: torch_cuda_path=torch._C._cuda_getLibPath(); print(torch_cuda_path if torch_cuda_path else ''); except: print('')")

if [ -n "$PYTORCH_CUDA_PATH" ]; then
    echo "Using PyTorch bundled CUDA libraries from: $PYTORCH_CUDA_PATH"
    export LD_LIBRARY_PATH=$PYTORCH_CUDA_PATH:$LD_LIBRARY_PATH
else
    echo "Warning: Could not get PyTorch bundled CUDA path. Falling back to PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK=1"
    export PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK=1
fi

cd tabular/
if [ -n "$ADDITIONAL_TEST_ARGS" ]
then
    python -m pytest --junitxml=results.xml --runslow "$ADDITIONAL_TEST_ARGS" tests
else
    python -m pytest --junitxml=results.xml --runslow tests
fi
