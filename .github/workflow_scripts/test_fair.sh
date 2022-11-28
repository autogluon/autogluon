#!/bin/bash

set -ex

ADDITIONAL_TEST_ARGS=$1

if [ -d fair ]
then
    source $(dirname "$0")/env_setup.sh

    setup_build_env
    export CUDA_VISIBLE_DEVICES=0
    install_core_all_tests
    install_common
    install_features
    install_tabular_all
    install_fair

    cd fair/
    if [ -n "$ADDITIONAL_TEST_ARGS" ]
    then
        python3 -m pytest --junitxml=results.xml --runslow "$ADDITIONAL_TEST_ARGS" tests
    else
        python3 -m pytest --junitxml=results.xml --runslow tests
    fi
fi
