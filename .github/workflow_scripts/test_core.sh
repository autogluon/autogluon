#!/bin/bash

set -ex

ADDITIONAL_TEST_ARGS=$1

source $(dirname "$0")/env_setup.sh

setup_build_env
install_local_packages "common/[tests]" "core/[all,tests]"

cd core/
if [ "$OSTYPE" == "msys" ]
then
    # to skip certain tests on Windows platform
    python3 -m pytest --junitxml=results.xml --runslow tests
elif [ -n "$ADDITIONAL_TEST_ARGS" ]
then
    python3 -m pytest --junitxml=results.xml --runslow --runplatform "$ADDITIONAL_TEST_ARGS" tests
else
    python3 -m pytest --junitxml=results.xml --runslow --runplatform tests
fi
