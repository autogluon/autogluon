#!/bin/bash

set -ex

ADDITIONAL_TEST_ARGS=$1

source $(dirname "$0")/env_setup.sh

setup_build_env
install_local_packages "common/[tests]" "core/[all,tests]"
python3 -m pip install ray_lightning==0.2.0  # TODO Change this line once we support ray_lightning 0.3.0

cd core/
if [ -n "$ADDITIONAL_TEST_ARGS" ]
then
    python3 -m pytest --junitxml=results.xml --runslow "$ADDITIONAL_TEST_ARGS" tests
else
    python3 -m pytest --junitxml=results.xml --runslow tests
fi
