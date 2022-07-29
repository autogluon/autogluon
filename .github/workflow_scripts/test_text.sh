#!/bin/bash

set -ex

ADDITIONAL_TEST_ARGS=$1

source $(dirname "$0")/env_setup.sh

setup_build_env
export CUDA_VISIBLE_DEVICES=0
install_core_all_tests
install_features
install_multimodal
install_text

cd text/
python3 -m pytest --junitxml=results.xml --runslow "$ADDITIONAL_TEST_ARGS" tests
