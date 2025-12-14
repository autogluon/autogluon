#!/bin/bash

set -ex

ADDITIONAL_TEST_ARGS=$1

source $(dirname "$0")/env_setup.sh

setup_build_env
install_local_packages "common/[tests]" "features/" "core/[all,tests]"

cd core/
if [ "$OSTYPE" == "msys" ]
then
    # to skip certain tests on Windows platform
    python -m pytest --junitxml=results.xml --runslow tests
elif [ -n "$ADDITIONAL_TEST_ARGS" ]
then
    python -m pytest --junitxml=results.xml --runslow --runplatform "$ADDITIONAL_TEST_ARGS" tests
else
    python -m pytest --junitxml=results.xml --runslow --runplatform tests
fi
