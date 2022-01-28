#!/bin/bash

set -ex

source $(dirname "$0")/env_setup.sh

setup_build_env
setup_mxnet_gpu
export CUDA_VISIBLE_DEVICES=0
install_core_all
install_vision

cd vision/
python3 -m pytest --junitxml=results.xml --runslow tests
