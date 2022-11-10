#!/bin/bash

set -ex

ADDITIONAL_TEST_ARGS=$1

source $(dirname "$0")/env_setup.sh

setup_build_env
install_common
install_features

cd features/
if [ -z "$ADDITIONAL_TEST_ARGS" ]
then
    python3 -m pytest --junitxml=results.xml --runslow "$ADDITIONAL_TEST_ARGS" tests
else
    python3 -m pytest --junitxml=results.xml --runslow tests
fi
