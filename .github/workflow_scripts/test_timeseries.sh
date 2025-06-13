#!/bin/bash

set -ex

while getopts ":-:" opt; do
  [ "$opt" = "-" ] && [ "${OPTARG}" = "is-platform-test" ] && IS_PLATFORM_TEST=1
done

source $(dirname "$0")/env_setup.sh

setup_build_env
export CUDA_VISIBLE_DEVICES=0
install_local_packages "common/[tests]" "core/[all,tests]" "features/" "tabular/[all,tests]" "timeseries/[all,tests]"
python -m pip install --upgrade pytest-xdist

export PYTHONHASHSEED=0  # for consistency in xdist tests

# Use wheel bundled CUDA instead of DLC CUDA 
export LD_LIBRARY_PATH=$(python -c "import torch; torch_cuda_path=''; try: torch_cuda_path=torch._C._cuda_getLibPath(); print(torch_cuda_path) if torch_cuda_path else ''; except: print('')"):$LD_LIBRARY_PATH:
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
nvidia-smi
# echo all exports
env

cd timeseries/
if [ "$IS_PLATFORM_TEST" = 1 ]; then
    python -m pytest --junitxml=results.xml --runslow tests  # run platform tests without multiprocessing
else
    python -m pytest --junitxml=results.xml --runslow --numprocesses 4 tests
fi
