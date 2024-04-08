#!/bin/bash

set -ex

ADDITIONAL_TEST_ARGS=$1

source $(dirname "$0")/env_setup.sh

setup_build_env
export CUDA_VISIBLE_DEVICES=0
install_local_packages "common/[tests]" "core/[all,tests]" "features/" "tabular/[all,tests]" "timeseries/[all,tests]"

cd timeseries/
if [ -n "$ADDITIONAL_TEST_ARGS" ]
then
    python -m pytest --junitxml=results.xml --runslow "$ADDITIONAL_TEST_ARGS" tests
else
    python -m pytest --junitxml=results.xml --runslow tests
fi
