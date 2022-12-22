#!/bin/bash

set -ex

ADDITIONAL_TEST_ARGS=$1

source $(dirname "$0")/env_setup.sh

setup_build_env
setup_torch_gpu
export CUDA_VISIBLE_DEVICES=0
install_local_packages "common/[tests]" "core/[all,tests]" "features/"
install_multimodal "[tests]"

cd multimodal/
if [ -n "$ADDITIONAL_TEST_ARGS" ]
then
    python3 -m pytest -n=2 -x --junitxml=results.xml --runslow "$ADDITIONAL_TEST_ARGS" tests/unittests/others/
else
    python3 -m pytest -n=2 -x --junitxml=results.xml --runslow tests/unittests/others/
fi
