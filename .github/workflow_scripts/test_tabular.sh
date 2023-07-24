#!/bin/bash

set -ex

ADDITIONAL_TEST_ARGS=$1
IS_PLATFORM_TEST=$2

source $(dirname "$0")/env_setup.sh

setup_build_env

if ! [ "$IS_PLATFORM_TEST" = "true" ]
then
    export CUDA_VISIBLE_DEVICES=0
fi

install_local_packages "common/[tests]" "core/[all,tests]" "features/"

if [ "$IS_PLATFORM_TEST" = "true" ]
then
    install_tabular_platforms "[all,tests]"
    install_multimodal_no_groundingdino "[tests]"
else
    install_tabular "[all,tests]"
    install_multimodal "[tests]"
fi

cd tabular/
if [ -n "$ADDITIONAL_TEST_ARGS" ]
then
    python3 -m pytest --junitxml=results.xml --runslow "$ADDITIONAL_TEST_ARGS" tests
else
    python3 -m pytest --junitxml=results.xml --runslow tests
fi
