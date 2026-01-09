#!/bin/bash

set -ex

ADDITIONAL_TEST_ARGS=$1

source $(dirname "$0")/env_setup.sh

setup_build_env
export CUDA_VISIBLE_DEVICES=0
install_local_packages "common/[tests]" "features/" "core/[all,tests]" "tabular/[all,tests]" "eda/[tests]"

cd eda/
python -m tox -e lint,typecheck,format,testenv
if [ -n "$ADDITIONAL_TEST_ARGS" ]
then
    python -m pytest --junitxml=results.xml --runslow "$ADDITIONAL_TEST_ARGS" tests
else
    python -m pytest --junitxml=results.xml --runslow tests
fi
