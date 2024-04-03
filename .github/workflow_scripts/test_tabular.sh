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
    install_multimodal "[tests]"
else
    install_tabular "[all,tests]"
    install_multimodal "[tests]"
fi

cd tabular/
if [ -n "$ADDITIONAL_TEST_ARGS" ]
then
    # TODO: Temporary re-arrangement of tests to unblock timeout on platform tests
    python -m pytest --junitxml=results.xml --runslow "$ADDITIONAL_TEST_ARGS" tests/regressiontests tests/unittests/calibrate tests/unittests/configs tests/unittests/data tests/unittests/dynamic_stacking tests/unittests/edgecases tests/unittests/experimental tests/unittests/models tests/unittests/pseudolabel tests/unittests/test_tabular.py tests/test_check_style.py tests/unittests/resource_allocation
else
    # TODO: Temporary re-arrangement of tests to unblock timeout on platform tests
    python -m pytest --junitxml=results.xml --runslow tests/regressiontests tests/unittests/calibrate tests/unittests/configs tests/unittests/data tests/unittests/dynamic_stacking tests/unittests/edgecases tests/unittests/experimental tests/unittests/models tests/unittests/pseudolabel tests/unittests/test_tabular.py tests/test_check_style.py tests/unittests/resource_allocation
fi

