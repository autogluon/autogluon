#!/bin/bash

set -ex

ADDITIONAL_TEST_ARGS=$1

source $(dirname "$0")/env_setup.sh

setup_build_env
setup_mxnet_gpu
setup_torch_gpu
export CUDA_VISIBLE_DEVICES=0
install_core_all_tests
install_vision

cd vision/
python3 -m pytest --junitxml=results.xml --runslow "$ADDITIONAL_TEST_ARGS" tests
