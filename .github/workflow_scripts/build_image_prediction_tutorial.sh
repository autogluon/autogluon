#!/bin/bash

set -ex

PR_NUMBER=$(basename $1) # For push events, this will be master branch instead of PR number
COMMIT_SHA=$2

source $(dirname "$0")/env_setup.sh

setup_build_contrib_env
setup_mxnet_gpu
setup_torch
export CUDA_VISIBLE_DEVICES=0
bash docs/build_pip_install.sh
# only build for docs/image_prediction
shopt -s extglob
rm -rf ./docs/tutorials/!(image_prediction)
cd docs && rm -rf _build && d2lbook build rst

COMMAND_EXIT_CODE=$?
if [ $COMMAND_EXIT_CODE -ne 0 ]; then
    exit COMMAND_EXIT_CODE
fi

cd ..
aws s3 cp --recursive docs/_build/rst/tutorials/image_prediction/ s3://autogluon-dev/build_docs/$PR_NUMBER/$COMMIT_SHA/image_prediction/ --quiet
