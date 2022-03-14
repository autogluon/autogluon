#!/bin/bash

set -ex

PR_NUMBER=$(basename $1) # For push events, this will be master branch instead of PR number
COMMIT_SHA=$2

source $(dirname "$0")/env_setup.sh

setup_build_contrib_env
setup_mxnet_gpu
export CUDA_VISIBLE_DEVICES=0
bash docs/build_pip_install.sh
# only build for docs/forecasting
shopt -s extglob
rm -rf ./docs/tutorials/!(forecasting)
cd docs && rm -rf _build && d2lbook build rst

COMMAND_EXIT_CODE=$?
if [ $COMMAND_EXIT_CODE -ne 0 ]; then
    exit COMMAND_EXIT_CODE
fi

cd ..
# Verify we still own the bucket
bucket_query=$(aws s3 ls | grep -E "(^| )autogluon-ci( |$)")
if [ -z bucket_query ]; then
    aws s3 cp --recursive docs/_build/rst/tutorials/forecasting/ s3://autogluon-ci/build_docs/$PR_NUMBER/$COMMIT_SHA/forecasting/ --quiet
else
    echo Bucket does not belong to us anymore. Will not write to it
fi;
