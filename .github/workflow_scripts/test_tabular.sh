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

install_local_packages "common/[tests]" "features/" "core/[all,tests]"

if [ "$IS_PLATFORM_TEST" = "true" ]
then
    install_tabular_platforms "[all,tests]"
    install_multimodal "[tests]"
else
    install_tabular "[all,tests]"
    install_multimodal "[tests]"
fi

if [[ "$(uname)" == "Darwin" ]]; then
    # On macOS torch and lightgbm each load a separate libomp, which segfaults.
    # Point the loader at torch's bundled libomp so both use the same runtime.
    export DYLD_LIBRARY_PATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

cd tabular/
if [ -n "$ADDITIONAL_TEST_ARGS" ]
then
    python -m pytest --junitxml=results.xml --runslow "$ADDITIONAL_TEST_ARGS" tests
else
    python -m pytest --junitxml=results.xml --runslow tests
fi
