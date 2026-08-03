#!/bin/bash

set -ex

while getopts ":-:" opt; do
  [ "$opt" = "-" ] && [ "${OPTARG}" = "is-platform-test" ] && IS_PLATFORM_TEST=1
done

source $(dirname "$0")/env_setup.sh

setup_build_env
export CUDA_VISIBLE_DEVICES=0
install_local_packages "common/[tests]" "features/" "core/[all,tests]" "tabular/[all,tests]" "timeseries/[all,tests]"
python -m pip install --upgrade pytest-xdist

export PYTHONHASHSEED=0  # for consistency in xdist tests
unset LD_LIBRARY_PATH  # avoid cuDNN version conflicts with PyTorch's bundled cuDNN

if [[ "$(uname)" == "Darwin" ]]; then
    # On macOS torch and lightgbm each load a separate libomp, which segfaults.
    # Point the loader at torch's bundled libomp so both use the same runtime.
    export DYLD_LIBRARY_PATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

cd timeseries/
if [ "$IS_PLATFORM_TEST" = 1 ]; then
    python -m pytest --junitxml=results.xml --runslow tests  # run platform tests without multiprocessing
else
    python -m pytest --junitxml=results.xml --runslow --numprocesses 4 tests
fi
